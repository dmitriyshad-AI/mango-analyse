from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.utils.phone import normalize_phone


AMO_MANUAL_RESOLUTION_SCHEMA_VERSION = "amo_manual_resolution_v1"
DEFAULT_REVIEW_BUCKETS = (
    "needs_manager_review_multi_contact",
    "blocked_contact_id_mismatch",
    "needs_text_quality_review",
)
ACCEPTED_STATUSES = {"accepted", "accepted_by_operator", "accepted_by_manager", "accepted_auto_policy"}
BLOCKED_STATUSES = {"blocked", "rejected", "needs_human", "needs_manager", "needs_text_review", "already_written_review"}


RESOLUTION_COLUMNS = [
    "resolution_id",
    "resolution_status",
    "resolved_contact_id",
    "allow_contact_id_outside_source",
    "resolution_reason",
    "resolved_by",
    "resolution_notes",
    "suggested_resolution_status",
    "suggested_resolved_contact_id",
    "suggested_reason",
    "queue_bucket",
    "queue_reason",
    "source_row_index",
    "phone",
    "source_amo_contact_ids",
    "dry_run_contact_ids",
    "written_status",
    "written_contact_id",
    "latest_call_date",
    "latest_call_type",
    "priority",
    "sale_probability_percent",
    "next_step",
    "fio_parent",
    "fio_child",
    "amo_lead_ids",
]


@dataclass(frozen=True)
class AmoManualResolutionSummary:
    schema_version: str
    generated_at: str
    queue_root: str
    source_csv: str
    out_root: str
    review_rows: int
    accepted_rows: int
    resolved_live_candidate_rows: int
    still_blocked_rows: int
    needs_human_rows: int
    already_written_review_rows: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_amo_manual_resolution_pack(
    *,
    queue_root: Path,
    source_csv: Path,
    out_root: Path,
    decisions_csv: Optional[Path] = None,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    """Build a fail-closed resolution pack for ambiguous AMO writeback rows.

    This command never writes to AMO. Accepted decisions only create a candidate
    CSV for a later dry-run/live stage. All rows without explicit accepted
    decisions remain blocked or needs-human.
    """

    queue_root = queue_root.expanduser().resolve(strict=False)
    source_csv = source_csv.expanduser().resolve(strict=False)
    out_root = out_root.expanduser().resolve(strict=False)
    out_root.mkdir(parents=True, exist_ok=True)
    now = generated_at or datetime.now(timezone.utc)

    source_rows = _read_csv(source_csv)
    source_by_index = {str(index): row for index, row in enumerate(source_rows, start=1)}
    source_by_phone = {normalize_phone(row.get("Телефон клиента")) or "": row for row in source_rows}
    review_rows = _load_review_queue_rows(queue_root)
    suggestions = [_build_suggestion(row) for row in review_rows]
    decisions = _load_decisions(decisions_csv) if decisions_csv else {}

    resolved_candidates: list[dict[str, Any]] = []
    still_blocked: list[dict[str, Any]] = []
    needs_human: list[dict[str, Any]] = []
    already_written_review: list[dict[str, Any]] = []
    applied_rows: list[dict[str, Any]] = []

    for suggestion in suggestions:
        decision = {**suggestion, **decisions.get(str(suggestion["resolution_id"]), {})}
        normalized = _normalize_decision(decision)
        applied_rows.append(normalized)
        status = str(normalized.get("resolution_status") or "").strip()
        if status in ACCEPTED_STATUSES:
            source = source_by_index.get(str(normalized.get("source_row_index") or "")) or source_by_phone.get(str(normalized.get("phone") or ""))
            candidate, error = _candidate_from_accepted_decision(source, normalized)
            if candidate is None:
                blocked = {**normalized, "validation_error": error}
                still_blocked.append(blocked)
            else:
                resolved_candidates.append(candidate)
        elif str(normalized.get("written_status") or "").strip().casefold() == "written":
            already_written_review.append(normalized)
        elif status in {"needs_human", "needs_manager", "needs_text_review"}:
            needs_human.append(normalized)
        else:
            still_blocked.append(normalized)

    template_path = out_root / "resolution_template.csv"
    default_decisions_path = out_root / "resolution_decisions_default.csv"
    applied_path = out_root / "resolution_decisions_applied.csv"
    candidates_path = out_root / "resolved_live_candidates_ru.csv"
    still_blocked_path = out_root / "still_blocked.csv"
    needs_human_path = out_root / "needs_human.csv"
    already_written_path = out_root / "already_written_review.csv"
    command_path = out_root / "next_dry_run_command.sh"

    _write_csv(template_path, suggestions, fieldnames=RESOLUTION_COLUMNS)
    _write_csv(default_decisions_path, [_default_decision_row(row) for row in suggestions], fieldnames=RESOLUTION_COLUMNS)
    _write_csv(applied_path, applied_rows, fieldnames=RESOLUTION_COLUMNS + ["validation_error"])
    _write_csv(candidates_path, resolved_candidates, fieldnames=_source_headers(source_rows))
    _write_csv(still_blocked_path, still_blocked, fieldnames=RESOLUTION_COLUMNS + ["validation_error"])
    _write_csv(needs_human_path, needs_human, fieldnames=RESOLUTION_COLUMNS + ["validation_error"])
    _write_csv(already_written_path, already_written_review, fieldnames=RESOLUTION_COLUMNS + ["validation_error"])

    command_path.write_text(_render_dry_run_command(candidates_path, out_root), encoding="utf-8")
    command_path.chmod(0o755)

    validation_errors = [row for row in still_blocked if row.get("validation_error")]
    accepted_rows = sum(1 for row in applied_rows if row.get("resolution_status") in ACCEPTED_STATUSES)
    status_counts = Counter(str(row.get("resolution_status") or "") for row in applied_rows)
    bucket_counts = Counter(str(row.get("queue_bucket") or "") for row in applied_rows)
    summary = AmoManualResolutionSummary(
        schema_version=AMO_MANUAL_RESOLUTION_SCHEMA_VERSION,
        generated_at=now.isoformat(timespec="seconds"),
        queue_root=str(queue_root),
        source_csv=str(source_csv),
        out_root=str(out_root),
        review_rows=len(review_rows),
        accepted_rows=accepted_rows,
        resolved_live_candidate_rows=len(resolved_candidates),
        still_blocked_rows=len(still_blocked),
        needs_human_rows=len(needs_human),
        already_written_review_rows=len(already_written_review),
        validation_ok=len(validation_errors) == 0,
        blocked=len(validation_errors),
        warnings=len(needs_human) + len(already_written_review),
    )
    result: dict[str, Any] = {
        "summary": summary.to_json_dict(),
        "counts": {
            "by_resolution_status": dict(status_counts),
            "by_queue_bucket": dict(bucket_counts),
        },
        "outputs": {
            "resolution_template_csv": str(template_path),
            "resolution_decisions_default_csv": str(default_decisions_path),
            "resolution_decisions_applied_csv": str(applied_path),
            "resolved_live_candidates_csv": str(candidates_path),
            "still_blocked_csv": str(still_blocked_path),
            "needs_human_csv": str(needs_human_path),
            "already_written_review_csv": str(already_written_path),
            "next_dry_run_command_sh": str(command_path),
            "summary_json": str(out_root / "summary.json"),
        },
        "policy": {
            "fail_closed": True,
            "unaccepted_rows_can_write_live": False,
            "accepted_rows_require_real_tunnel_dry_run": True,
            "accepted_rows_require_quality_gate_for_live": True,
            "live_write_executed": False,
        },
        "next_actions": _next_actions(len(resolved_candidates), len(needs_human), len(already_written_review), len(still_blocked)),
    }
    (out_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_root / "README.md").write_text(_render_readme(result), encoding="utf-8")
    return result


def _load_review_queue_rows(queue_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for bucket in DEFAULT_REVIEW_BUCKETS:
        path = queue_root / f"{bucket}.csv"
        for row in _read_csv(path):
            row = dict(row)
            row.setdefault("queue_bucket", bucket)
            rows.append(row)
    return rows


def _build_suggestion(row: Mapping[str, Any]) -> dict[str, Any]:
    bucket = _safe_text(row.get("queue_bucket"))
    phone = normalize_phone(row.get("normalized_phone") or row.get("Телефон клиента")) or _safe_text(row.get("normalized_phone") or row.get("Телефон клиента"))
    source_ids = _split_ids(row.get("source_amo_contact_ids") or row.get("AMO contact IDs"))
    dry_ids = _split_ids(row.get("dry_run_contact_id"))
    written_status = _safe_text(row.get("written_status"))
    suggested_status = "needs_human"
    suggested_contact = ""
    suggested_reason = "manual_confirmation_required"
    if bucket == "needs_manager_review_multi_contact":
        if len(source_ids) == 1 and source_ids[0] in dry_ids:
            suggested_status = "suggest_source_contact_id_for_manager_confirmation"
            suggested_contact = source_ids[0]
            suggested_reason = "source_amo_contact_id_is_present_in_live_exact_matches"
    elif bucket == "blocked_contact_id_mismatch":
        suggested_status = "blocked_contact_id_mismatch"
        suggested_contact = source_ids[0] if len(source_ids) == 1 else ""
        suggested_reason = "live_lookup_contact_id_differs_from_source_amo_contact_id"
    elif bucket == "needs_text_quality_review":
        if written_status.casefold() == "written":
            suggested_status = "already_written_review"
            suggested_reason = "row_was_written_before_but_now_is_review_marked"
        else:
            suggested_status = "needs_text_review"
            suggested_reason = "crm_text_review_marker_blocks_live_write"
        suggested_contact = source_ids[0] if len(source_ids) == 1 else ""
    resolution_id = f"row{_safe_text(row.get('source_row_index')).zfill(4)}_{phone.replace('+', '')}"
    return {
        "resolution_id": resolution_id,
        "resolution_status": "needs_human",
        "resolved_contact_id": "",
        "allow_contact_id_outside_source": "no",
        "resolution_reason": "",
        "resolved_by": "",
        "resolution_notes": "",
        "suggested_resolution_status": suggested_status,
        "suggested_resolved_contact_id": suggested_contact,
        "suggested_reason": suggested_reason,
        "queue_bucket": bucket,
        "queue_reason": _safe_text(row.get("queue_reason")),
        "source_row_index": _safe_text(row.get("source_row_index")),
        "phone": phone,
        "source_amo_contact_ids": " | ".join(source_ids),
        "dry_run_contact_ids": " | ".join(dry_ids),
        "written_status": written_status,
        "written_contact_id": _safe_text(row.get("written_contact_id")),
        "latest_call_date": _safe_text(row.get("Дата последнего свежего звонка")),
        "latest_call_type": _safe_text(row.get("Тип последнего свежего звонка")),
        "priority": _safe_text(row.get("Приоритет лида")),
        "sale_probability_percent": _safe_text(row.get("Вероятность продажи, %")),
        "next_step": _safe_text(row.get("Следующий шаг")),
        "fio_parent": _safe_text(row.get("ФИО родителя")),
        "fio_child": _safe_text(row.get("ФИО ребенка")),
        "amo_lead_ids": _safe_text(row.get("AMO lead IDs")),
    }


def _default_decision_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    bucket = _safe_text(result.get("queue_bucket"))
    if bucket == "blocked_contact_id_mismatch":
        result["resolution_status"] = "blocked"
        result["resolution_reason"] = "contact_id_mismatch_requires_operator_check"
    elif bucket == "needs_text_quality_review":
        if _safe_text(result.get("written_status")).casefold() == "written":
            result["resolution_status"] = "already_written_review"
            result["resolution_reason"] = "do_not_rewrite_until_text_review_is_resolved"
        else:
            result["resolution_status"] = "needs_text_review"
            result["resolution_reason"] = "crm_text_requires_review_before_live_write"
    else:
        result["resolution_status"] = "needs_manager"
        result["resolution_reason"] = "multiple_exact_amo_contacts_require_manager_choice"
    return result


def _normalize_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    row = {key: _safe_text(decision.get(key)) for key in RESOLUTION_COLUMNS}
    status = row["resolution_status"].casefold() or "needs_human"
    aliases = {
        "accept": "accepted",
        "accepted_manager": "accepted_by_manager",
        "accepted_operator": "accepted_by_operator",
        "manual": "needs_human",
        "needs_manual": "needs_human",
        "text_review": "needs_text_review",
    }
    row["resolution_status"] = aliases.get(status, status)
    return row


def _candidate_from_accepted_decision(source: Optional[Mapping[str, Any]], decision: Mapping[str, Any]) -> tuple[Optional[dict[str, Any]], str]:
    if source is None:
        return None, "source_row_not_found"
    resolved = _safe_text(decision.get("resolved_contact_id"))
    if not resolved.isdigit():
        return None, "accepted_resolution_requires_numeric_resolved_contact_id"
    resolution_reason = _safe_text(decision.get("resolution_reason")).casefold()
    resolved_by = _safe_text(decision.get("resolved_by"))
    if not resolution_reason:
        return None, "accepted_resolution_requires_reason"
    if not resolved_by:
        return None, "accepted_resolution_requires_resolved_by"
    bucket = _safe_text(decision.get("queue_bucket"))
    written_status = _safe_text(decision.get("written_status")).casefold()
    if bucket == "needs_manager_review_multi_contact" and "post_merge_recheck_approved" not in resolution_reason:
        return None, "duplicate_merge_requires_post_merge_recheck_approved_reason"
    if bucket == "needs_text_quality_review":
        if "text_quality_approved" not in resolution_reason:
            return None, "text_quality_review_requires_explicit_approval_reason"
        if written_status == "written" and "refresh_approved" not in resolution_reason:
            return None, "already_written_review_requires_refresh_approved_reason"
    source_ids = set(_split_ids(decision.get("source_amo_contact_ids") or source.get("AMO contact IDs")))
    dry_ids = set(_split_ids(decision.get("dry_run_contact_ids")))
    allow_outside = _safe_text(decision.get("allow_contact_id_outside_source")).casefold() in {"yes", "true", "1", "да"}
    if allow_outside and "outside_source_approved" not in resolution_reason:
        return None, "outside_source_override_requires_explicit_approval_reason"
    if source_ids and resolved not in source_ids and not allow_outside:
        return None, "resolved_contact_id_not_in_source_amo_contact_ids"
    if dry_ids and resolved not in dry_ids and not allow_outside:
        return None, "resolved_contact_id_not_in_dry_run_contact_ids"
    candidate = dict(source)
    candidate["AMO contact IDs"] = resolved
    candidate["Manual resolution ID"] = _safe_text(decision.get("resolution_id"))
    candidate["Manual resolution status"] = _safe_text(decision.get("resolution_status"))
    candidate["Manual resolution reason"] = _safe_text(decision.get("resolution_reason"))
    candidate["Manual resolution source bucket"] = _safe_text(decision.get("queue_bucket"))
    candidate["Готово к записи в AMO"] = "Да"
    reason = _safe_text(candidate.get("Причина статуса AMO"))
    candidate["Причина статуса AMO"] = (reason + " | manual resolution accepted").strip(" |")
    return candidate, ""


def _load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for row in _read_csv(path.expanduser().resolve(strict=False)):
        resolution_id = _safe_text(row.get("resolution_id"))
        if resolution_id:
            decisions[resolution_id] = dict(row)
    return decisions


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: list[Mapping[str, Any]], *, fieldnames: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify(row.get(key, "")) for key in fieldnames})


def _source_headers(rows: list[Mapping[str, Any]]) -> list[str]:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    for extra in ("Manual resolution ID", "Manual resolution status", "Manual resolution reason", "Manual resolution source bucket"):
        if extra not in headers:
            headers.append(extra)
    return headers


def _render_dry_run_command(candidates_path: Path, out_root: Path) -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
CANDIDATES="{candidates_path}"
if ! awk 'NR > 1 {{ found=1; exit }} END {{ exit found ? 0 : 1 }}' "$CANDIDATES"; then
  echo "No resolved live candidates. Dry-run is intentionally skipped."
  exit 0
fi
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/private/tmp/uv-cache uv run \
  --with pandas --with openpyxl --with xlsxwriter \
  --with sqlalchemy --with requests --with 'psycopg[binary]' \
  python scripts/write_amo_ready_contacts.py \
  --input "$CANDIDATES" \
  --crm-writeback-quality-summary "{out_root / 'crm_quality_gate' / 'summary.json'}" \
  --quality-gate-summary "stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json"
'''


def _render_readme(result: Mapping[str, Any]) -> str:
    summary = _mapping(result.get("summary"))
    return f"""# AMO manual-resolution pack

Generated at: `{summary.get('generated_at')}`

This pack is fail-closed. It does not write to AMO and does not authorize live writeback.

Rows:

- Review rows: `{summary.get('review_rows')}`
- Accepted rows: `{summary.get('accepted_rows')}`
- Resolved live candidates: `{summary.get('resolved_live_candidate_rows')}`
- Still blocked: `{summary.get('still_blocked_rows')}`
- Needs human: `{summary.get('needs_human_rows')}`
- Already-written review: `{summary.get('already_written_review_rows')}`

Use `resolution_template.csv` for operator decisions. Only rows with `resolution_status` in `{', '.join(sorted(ACCEPTED_STATUSES))}` can enter `resolved_live_candidates_ru.csv`.

No live writeback may happen until resolved candidates pass CRM quality gate, real-tunnel dry-run, explicit live confirmation and post-writeback readback.
"""


def _next_actions(candidates: int, needs_human: int, already_written_review: int, still_blocked: int) -> list[Mapping[str, Any]]:
    actions: list[Mapping[str, Any]] = []
    if candidates:
        actions.append({"action": "run_quality_gate_on_resolved_candidates", "rows": candidates})
        actions.append({"action": "run_real_tunnel_dry_run_on_resolved_candidates", "rows": candidates})
    if needs_human:
        actions.append({"action": "fill_resolution_decisions_for_needs_human", "rows": needs_human})
    if already_written_review:
        actions.append({"action": "decide_refresh_or_leave_already_written_review_rows", "rows": already_written_review})
    if still_blocked:
        actions.append({"action": "investigate_still_blocked_rows", "rows": still_blocked})
    if not actions:
        actions.append({"action": "no_manual_resolution_work_remaining", "rows": 0})
    return actions


def _split_ids(value: Any) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    return [part for part in re.split(r"[|,;\s]+", text) if part.strip()]


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return _safe_text(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "AMO_MANUAL_RESOLUTION_SCHEMA_VERSION",
    "build_amo_manual_resolution_pack",
]
