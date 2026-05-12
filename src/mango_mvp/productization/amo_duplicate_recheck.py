from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.utils.phone import normalize_phone


AMO_DUPLICATE_RECHECK_SCHEMA_VERSION = "amo_duplicate_post_merge_recheck_v1"
DEFAULT_DUPLICATE_PACK_ROOT = Path("stable_runtime/amo_duplicate_resolution_20260511_v1")
DEFAULT_REPORTS_ROOT = Path("stable_runtime/amocrm_runtime/contact_writebacks")
DEFAULT_OUT_ROOT = Path("stable_runtime/amo_duplicate_post_merge_recheck_20260511_v1")

ROW_RESULT_COLUMNS = [
    "resolution_id",
    "phone",
    "expected_source_contact_ids",
    "candidate_contact_ids",
    "report_status",
    "report_reason",
    "report_contact_ids",
    "surviving_contact_id",
    "survivor_relation",
    "decision",
    "blocking_reason",
]

BLOCKING_REPORT_REASONS = {
    "multiple_exact_contacts_in_amo",
    "contact_id_mismatch_with_source_amo_contact_ids",
    "contact_not_found_in_amo",
    "invalid_phone",
    "empty_payload",
}


def build_amo_duplicate_post_merge_recheck(
    *,
    duplicate_pack_root: Path = DEFAULT_DUPLICATE_PACK_ROOT,
    report_dir: Optional[Path] = None,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    """Validate AMO post-merge dry-run results for duplicate-contact rows.

    This is intentionally read-only and fail-closed. It does not call AMO and it
    does not write CRM fields. It only evaluates an already produced dry-run
    report. If no matching dry-run report exists yet, the result is pending.
    """

    duplicate_pack_root = duplicate_pack_root.expanduser().resolve(strict=False)
    out_root = out_root.expanduser().resolve(strict=False)
    reports_root = reports_root.expanduser().resolve(strict=False)
    now = generated_at or datetime.now(timezone.utc)
    out_root.mkdir(parents=True, exist_ok=True)

    queue_rows = _read_csv(duplicate_pack_root / "duplicate_merge_queue.csv")
    recheck_rows = _read_csv(duplicate_pack_root / "post_merge_recheck_input_ru.csv")
    candidate_rows = _read_csv(duplicate_pack_root / "candidate_contacts.csv")
    candidate_ids_by_phone = _candidate_ids_by_phone(candidate_rows)
    expected_input = (duplicate_pack_root / "post_merge_recheck_input_ru.csv").expanduser().resolve(strict=False)
    report_dir = _resolve_report_dir(
        explicit_report_dir=report_dir,
        reports_root=reports_root,
        expected_input=expected_input,
    )

    report_summary: Mapping[str, Any] = {}
    report_rows: list[dict[str, str]] = []
    if report_dir:
        report_summary = _read_json(report_dir / "contact_writeback_summary.json")
        report_rows = _read_csv(report_dir / "contact_writeback_report.csv")

    report_by_phone: dict[str, list[dict[str, str]]] = {}
    for row in report_rows:
        phone = normalize_phone(_first_value(row, ["phone", "Телефон клиента", "normalized_phone"])) or _safe_text(
            _first_value(row, ["phone", "Телефон клиента", "normalized_phone"])
        )
        if phone:
            report_by_phone.setdefault(phone, []).append(row)

    row_results: list[dict[str, str]] = []
    for expected in recheck_rows:
        phone = normalize_phone(expected.get("Телефон клиента")) or _safe_text(expected.get("Телефон клиента"))
        matches = report_by_phone.get(phone, [])
        row_results.append(
            _evaluate_expected_row(
                expected=expected,
                matches=matches,
                candidate_ids=candidate_ids_by_phone.get(phone, []),
                report_summary=report_summary,
            )
        )

    summary_status, passed, blockers = _summarize_status(
        row_results=row_results,
        expected_rows=len(recheck_rows),
        expected_input=expected_input,
        report_dir=report_dir,
        report_summary=report_summary,
    )
    decision_counts = dict(Counter(row["decision"] for row in row_results))
    blocking_counts = dict(Counter(row["blocking_reason"] for row in row_results if row.get("blocking_reason")))

    outputs = {
        "row_results_csv": out_root / "row_results.csv",
        "summary_json": out_root / "summary.json",
        "readme_md": out_root / "README.md",
    }
    _write_csv(outputs["row_results_csv"], row_results, ROW_RESULT_COLUMNS)
    summary = {
        "schema_version": AMO_DUPLICATE_RECHECK_SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "duplicate_pack_root": str(duplicate_pack_root),
        "out_root": str(out_root),
        "report_dir": str(report_dir) if report_dir else "",
        "expected_input": str(expected_input),
        "report_input": _safe_text(report_summary.get("input")),
        "report_summary_path": str(report_dir / "contact_writeback_summary.json") if report_dir else "",
        "report_csv_path": str(report_dir / "contact_writeback_report.csv") if report_dir else "",
        "status": summary_status,
        "passed": passed,
        "expected_rows": len(recheck_rows),
        "queue_rows": len(queue_rows),
        "report_rows": len(report_rows),
        "ready_after_merge_rows": decision_counts.get("ready_after_merge", 0),
        "blocked_rows": len([row for row in row_results if row["decision"] != "ready_after_merge"]),
        "decision_counts": decision_counts,
        "blocking_counts": blocking_counts,
        "global_blockers": blockers,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "policy": {
            "read_only": True,
            "write_crm": False,
            "live_write_executed": False,
            "requires_real_tunnel_dry_run": True,
            "requires_one_surviving_contact_per_phone": True,
            "fail_closed": True,
        },
        "next_actions": _next_actions(summary_status, row_results),
    }
    _write_json(outputs["summary_json"], summary)
    outputs["readme_md"].write_text(_render_readme(summary), encoding="utf-8")
    return summary


def _resolve_report_dir(*, explicit_report_dir: Optional[Path], reports_root: Path, expected_input: Path) -> Optional[Path]:
    if explicit_report_dir:
        candidate = explicit_report_dir.expanduser().resolve(strict=False)
        return candidate if candidate.exists() else None
    expected = expected_input.expanduser().resolve(strict=False)
    candidates: list[Path] = []
    if reports_root.exists():
        for summary_path in reports_root.glob("*/contact_writeback_summary.json"):
            payload = _read_json(summary_path)
            input_path = _safe_text(payload.get("input"))
            if input_path and Path(input_path).expanduser().resolve(strict=False) == expected:
                candidates.append(summary_path.parent)
    return sorted(candidates)[-1] if candidates else None


def _evaluate_expected_row(
    *,
    expected: Mapping[str, Any],
    matches: list[Mapping[str, Any]],
    candidate_ids: list[str],
    report_summary: Mapping[str, Any],
) -> dict[str, str]:
    phone = normalize_phone(expected.get("Телефон клиента")) or _safe_text(expected.get("Телефон клиента"))
    resolution_id = _safe_text(expected.get("Manual resolution id"))
    source_ids = _split_ids(expected.get("AMO contact IDs"))
    candidate_ids = _unique([*candidate_ids, *source_ids])
    base = {
        "resolution_id": resolution_id,
        "phone": phone,
        "expected_source_contact_ids": " | ".join(source_ids),
        "candidate_contact_ids": " | ".join(candidate_ids),
        "report_status": "",
        "report_reason": "",
        "report_contact_ids": "",
        "surviving_contact_id": "",
        "survivor_relation": "",
        "decision": "blocked",
        "blocking_reason": "",
    }
    if not matches:
        return {**base, "decision": "pending", "blocking_reason": "missing_phone_in_dry_run_report"}
    if len(matches) > 1:
        contact_ids = _unique([contact_id for row in matches for contact_id in _split_ids(_first_value(row, ["contact_id", "AMO contact IDs"]))])
        return {
            **base,
            "report_status": " | ".join(_safe_text(row.get("status")) for row in matches),
            "report_reason": "multiple_report_rows_for_phone",
            "report_contact_ids": " | ".join(contact_ids),
            "decision": "blocked",
            "blocking_reason": "multiple_report_rows_for_phone",
        }
    report = matches[0]
    status = _safe_text(_first_value(report, ["status", "dry_run_status", "write_status"]))
    reason = _safe_text(_first_value(report, ["reason", "dry_run_reason", "skip_reason"]))
    report_contact_ids = _split_ids(_first_value(report, ["contact_id", "dry_run_contact_id", "effective_contact_id", "AMO contact IDs"]))
    result = {
        **base,
        "report_status": status,
        "report_reason": reason,
        "report_contact_ids": " | ".join(report_contact_ids),
        "surviving_contact_id": report_contact_ids[0] if len(report_contact_ids) == 1 else "",
    }
    if bool(report_summary.get("live_write")) or _safe_text(report_summary.get("mode")) == "live_write" or status == "written":
        return {**result, "decision": "blocked", "blocking_reason": "live_write_report_not_allowed_for_recheck"}
    if status != "dry_run":
        return {**result, "decision": "blocked", "blocking_reason": reason or f"unexpected_status:{status}"}
    if reason in BLOCKING_REPORT_REASONS:
        return {**result, "decision": "blocked", "blocking_reason": reason}
    if len(report_contact_ids) != 1:
        return {**result, "decision": "blocked", "blocking_reason": "not_exactly_one_surviving_contact_id"}
    surviving = report_contact_ids[0]
    if candidate_ids and surviving not in candidate_ids:
        return {**result, "survivor_relation": "outside_known_candidates", "decision": "needs_operator_review", "blocking_reason": "surviving_contact_id_outside_candidate_ids"}
    relation = "source_contact_id" if surviving in source_ids else "known_candidate_outside_source"
    return {**result, "survivor_relation": relation, "decision": "ready_after_merge", "blocking_reason": ""}


def _summarize_status(
    *,
    row_results: list[Mapping[str, str]],
    expected_rows: int,
    expected_input: Path,
    report_dir: Optional[Path],
    report_summary: Mapping[str, Any],
) -> tuple[str, bool, list[str]]:
    blockers: list[str] = []
    if expected_rows == 0:
        return "no_rows", True, []
    if not report_dir:
        return "pending_not_run", False, ["post_merge_real_tunnel_dry_run_missing"]
    if not report_summary:
        blockers.append("dry_run_summary_missing")
    report_input = _safe_text(report_summary.get("input"))
    if not report_input:
        blockers.append("dry_run_summary_input_missing")
    elif Path(report_input).expanduser().resolve(strict=False) != expected_input:
        blockers.append("dry_run_summary_input_mismatch")
    if int(report_summary.get("total_rows") or 0) != expected_rows:
        blockers.append("dry_run_total_rows_mismatch")
    if int(report_summary.get("dry_run") or 0) != expected_rows:
        blockers.append("dry_run_expected_count_mismatch")
    if bool(report_summary.get("preflight_failed")):
        blockers.append("dry_run_preflight_failed")
    if bool(report_summary.get("live_write")) or _safe_text(report_summary.get("mode")) == "live_write":
        blockers.append("live_write_report_not_allowed_for_recheck")
    if int(report_summary.get("failed") or 0) != 0:
        blockers.append("dry_run_failed_rows_present")
    not_ready = [row for row in row_results if row.get("decision") != "ready_after_merge"]
    if not_ready:
        blockers.append("row_level_recheck_not_green")
    passed = not blockers and len(row_results) == expected_rows
    return ("passed" if passed else "blocked"), passed, blockers


def _next_actions(status: str, rows: list[Mapping[str, str]]) -> list[Mapping[str, Any]]:
    if status == "passed":
        return [
            {
                "action": "prepare_bounded_next_stage_preflight",
                "rows": len(rows),
                "description_ru": "Все post-merge строки прошли dry-run recheck; можно готовить отдельный bounded preflight, но не broad live writeback.",
            }
        ]
    if status == "pending_not_run":
        return [
            {
                "action": "run_post_merge_real_tunnel_dry_run",
                "rows": len(rows),
                "description_ru": "После ручной склейки дублей запустить next_recheck_command.sh и затем повторить этот gate с report_dir dry-run.",
            }
        ]
    blocked = Counter(row.get("blocking_reason") or "unknown" for row in rows if row.get("decision") != "ready_after_merge")
    return [
        {
            "action": "resolve_remaining_duplicate_recheck_blockers",
            "rows": sum(blocked.values()),
            "blocking_counts": dict(blocked),
            "description_ru": "Оставить строки заблокированными, устранить причины и повторить post-merge recheck.",
        }
    ]


def _candidate_ids_by_phone(rows: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in rows:
        phone = normalize_phone(row.get("phone")) or _safe_text(row.get("phone"))
        contact_id = _safe_text(row.get("candidate_contact_id"))
        if phone and contact_id:
            result.setdefault(phone, [])
            if contact_id not in result[phone]:
                result[phone].append(contact_id)
    return result


def _first_value(row: Mapping[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if _safe_text(value):
            return value
    return ""


def _split_ids(value: Any) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    parts: list[str] = []
    for token in text.replace("|", ",").replace(";", ",").split(","):
        token = token.strip()
        if token:
            parts.append(token)
    if len(parts) == 1 and " " in parts[0]:
        parts = [part for part in parts[0].split() if part]
    return _unique(parts)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        value = _safe_text(value)
        if value and value not in result:
            result.append(value)
    return result


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: list[Mapping[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_readme(summary: Mapping[str, Any]) -> str:
    return f"""# AMO duplicate post-merge recheck gate

Read-only gate for duplicate-contact rows after manual AMO merge.

- Status: `{summary.get('status')}`
- Passed: `{summary.get('passed')}`
- Expected rows: `{summary.get('expected_rows')}`
- Ready after merge rows: `{summary.get('ready_after_merge_rows')}`
- Blocked rows: `{summary.get('blocked_rows')}`
- Report dir: `{summary.get('report_dir') or 'not found'}`

Rules:

1. The report must be a real-tunnel dry-run, not live writeback.
2. Every expected phone must be present in the dry-run report.
3. Each phone must resolve to exactly one AMO contact.
4. Rows with `multiple_exact_contacts_in_amo`, contact-id mismatch, missing contact or failed status stay blocked.
5. This gate never writes to AMO and never authorizes broad live writeback by itself.
"""
