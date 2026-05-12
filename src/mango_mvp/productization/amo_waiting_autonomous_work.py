from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.quality.crm_text_quality_detector import (
    detect_crm_text_quality_risks,
    has_blocking_crm_text_findings,
)
from mango_mvp.quality.crm_writeback_quality_detector import detect_crm_writeback_quality_risks
from mango_mvp.utils.phone import normalize_phone
from scripts.write_amo_ready_contacts import TARGET_CONTACT_FIELDS, _build_contact_payload


AMO_WAITING_AUTONOMOUS_WORK_SCHEMA_VERSION = "amo_waiting_autonomous_work_v1"
DEFAULT_QUEUE_ROOT = Path("stable_runtime/amo_writeback_queue_20260510_v2_production")
DEFAULT_OUT_ROOT = Path("stable_runtime/amo_waiting_autonomous_work_20260511_v1")
DEFAULT_CURRENT_RUNTIME_PATH = Path("stable_runtime/CURRENT_RUNTIME.json")
DEFAULT_CONTACT_WRITEBACK_REPORTS_ROOT = Path("stable_runtime/amocrm_runtime/contact_writebacks")
DEFAULT_STAGE15_SUMMARY = Path("stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json")
DEFAULT_FROZEN_CORPUS = Path("tests/fixtures/crm_writeback_relevance_frozen_corpus.jsonl")

CRM_TEXT_FIELDS = (
    "Краткое резюме последнего свежего звонка",
    "Краткая история общения",
    "Хронология общения (последние 5 касаний)",
    "Возражения",
    "Следующий шаг",
    "История общения Tallanto",
)


def build_amo_waiting_autonomous_work(
    *,
    project_root: Path,
    queue_root: Path = DEFAULT_QUEUE_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    current_runtime_path: Path = DEFAULT_CURRENT_RUNTIME_PATH,
    contact_writeback_reports_root: Path = DEFAULT_CONTACT_WRITEBACK_REPORTS_ROOT,
    stage15_summary: Path = DEFAULT_STAGE15_SUMMARY,
    frozen_corpus_jsonl: Path = DEFAULT_FROZEN_CORPUS,
    analysis_date: Optional[str] = None,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    """Prepare all safe work that can be done while AMO duplicates are being merged.

    This builder is intentionally read-only. It does not call AMO and it does not
    write CRM fields. It prepares bounded next batches and commands that still
    require quality gates, real-tunnel dry-runs and explicit live approval.
    """

    project_root = project_root.expanduser().resolve(strict=False)
    queue_root = _resolve(project_root, queue_root)
    out_root = _resolve(project_root, out_root)
    current_runtime_path = _resolve(project_root, current_runtime_path)
    contact_writeback_reports_root = _resolve(project_root, contact_writeback_reports_root)
    stage15_summary = _resolve(project_root, stage15_summary)
    frozen_corpus_jsonl = _resolve(project_root, frozen_corpus_jsonl)
    now = generated_at or datetime.now(timezone.utc)
    analysis_date = analysis_date or date.today().isoformat()
    out_root.mkdir(parents=True, exist_ok=True)

    runtime = _read_json(current_runtime_path)
    paths = _mapping(runtime.get("paths"))
    source_csv = _path_from_text(paths.get("amo_export_ready_csv"))
    active_export_summary = _read_json(_path_from_text(paths.get("active_export_summary")))
    tenant_config_path = _path_from_text(_mapping(active_export_summary.get("tenant_config")).get("path"))

    text_rows = _read_csv(queue_root / "needs_text_quality_review.csv")
    mismatch_rows = _read_csv(queue_root / "blocked_contact_id_mismatch.csv")
    already_written_rows = _read_csv(queue_root / "already_written.csv")

    text_report_rows, text_candidate_rows, text_refresh_rows = _evaluate_text_quality_rows(text_rows, analysis_date=analysis_date)
    mismatch_report_rows, mismatch_recheck_rows = _evaluate_mismatch_rows(mismatch_rows)
    readback_by_contact_id = _latest_readback_by_contact_id(contact_writeback_reports_root)
    refresh_diff_rows, refresh_candidates, readback_missing_rows = _build_refresh_diff(
        already_written_rows=already_written_rows,
        extra_written_rows=text_refresh_rows,
        readback_by_contact_id=readback_by_contact_id,
    )
    non_duplicate_candidates = text_candidate_rows
    refresh_candidates = _unique_candidates(refresh_candidates)

    outputs = {
        "summary_json": out_root / "summary.json",
        "dashboard_html": out_root / "dashboard.html",
        "non_duplicate_blockers_report_csv": out_root / "non_duplicate_blockers_report.csv",
        "text_quality_cleared_candidates_csv": out_root / "text_quality_cleared_candidates_ru.csv",
        "contact_id_mismatch_tasks_csv": out_root / "contact_id_mismatch_tasks.csv",
        "contact_id_mismatch_recheck_input_csv": out_root / "contact_id_mismatch_recheck_input_ru.csv",
        "already_written_refresh_diff_csv": out_root / "already_written_refresh_diff.csv",
        "already_written_refresh_candidates_csv": out_root / "already_written_refresh_candidates_ru.csv",
        "readback_missing_written_rows_csv": out_root / "readback_missing_written_rows.csv",
        "readback_missing_writeback_report_csv": out_root / "readback_missing_writeback_report.csv",
        "next_non_duplicate_quality_gate_command_sh": out_root / "next_non_duplicate_quality_gate_command.sh",
        "next_non_duplicate_real_tunnel_dry_run_command_sh": out_root / "next_non_duplicate_real_tunnel_dry_run_command.sh",
        "next_refresh_quality_gate_command_sh": out_root / "next_refresh_quality_gate_command.sh",
        "next_refresh_real_tunnel_dry_run_command_sh": out_root / "next_refresh_real_tunnel_dry_run_command.sh",
        "next_readback_missing_commands_sh": out_root / "next_readback_missing_commands.sh",
        "post_merge_full_after_staff_done_command_sh": out_root / "run_post_merge_full_after_staff_done.sh",
        "readme_md": out_root / "README.md",
    }

    source_headers = _read_headers(source_csv)
    _write_csv(outputs["non_duplicate_blockers_report_csv"], [*text_report_rows, *mismatch_report_rows])
    _write_csv(outputs["text_quality_cleared_candidates_csv"], non_duplicate_candidates, fieldnames=source_headers)
    _write_csv(outputs["contact_id_mismatch_tasks_csv"], mismatch_report_rows)
    _write_csv(outputs["contact_id_mismatch_recheck_input_csv"], mismatch_recheck_rows, fieldnames=source_headers)
    _write_csv(outputs["already_written_refresh_diff_csv"], refresh_diff_rows)
    _write_csv(outputs["already_written_refresh_candidates_csv"], refresh_candidates, fieldnames=source_headers)
    _write_csv(outputs["readback_missing_written_rows_csv"], readback_missing_rows)
    _write_csv(outputs["readback_missing_writeback_report_csv"], _missing_readback_report_rows(readback_missing_rows))

    _write_quality_and_dry_run_commands(
        project_root=project_root,
        candidate_csv=outputs["text_quality_cleared_candidates_csv"],
        quality_out=out_root / "non_duplicate_quality_gate",
        quality_script=outputs["next_non_duplicate_quality_gate_command_sh"],
        dry_run_script=outputs["next_non_duplicate_real_tunnel_dry_run_command_sh"],
        stage15_summary=stage15_summary,
        frozen_corpus_jsonl=frozen_corpus_jsonl,
        tenant_config_path=tenant_config_path,
        analysis_date=analysis_date,
        rows=len(non_duplicate_candidates),
    )
    _write_quality_and_dry_run_commands(
        project_root=project_root,
        candidate_csv=outputs["already_written_refresh_candidates_csv"],
        quality_out=out_root / "refresh_quality_gate",
        quality_script=outputs["next_refresh_quality_gate_command_sh"],
        dry_run_script=outputs["next_refresh_real_tunnel_dry_run_command_sh"],
        stage15_summary=stage15_summary,
        frozen_corpus_jsonl=frozen_corpus_jsonl,
        tenant_config_path=tenant_config_path,
        analysis_date=analysis_date,
        rows=len(refresh_candidates),
    )
    _write_readback_missing_command(
        outputs["next_readback_missing_commands_sh"],
        project_root=project_root,
        report_csv=outputs["readback_missing_writeback_report_csv"],
        out_root=out_root / "readback_missing_gate",
        rows=len(readback_missing_rows),
    )
    _write_post_merge_full_command(outputs["post_merge_full_after_staff_done_command_sh"], project_root=project_root)

    status = _status(
        text_candidates=len(non_duplicate_candidates),
        refresh_candidates=len(refresh_candidates),
        readback_missing=len(readback_missing_rows),
        mismatch=len(mismatch_report_rows),
    )
    summary = {
        "schema_version": AMO_WAITING_AUTONOMOUS_WORK_SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "queue_root": str(queue_root),
        "source_csv": str(source_csv) if source_csv else "",
        "out_root": str(out_root),
        "status": status,
        "counts": {
            "text_quality_review_rows": len(text_rows),
            "text_quality_cleared_rows": sum(1 for row in text_report_rows if _safe_text(row.get("decision")).startswith("text_quality_cleared")),
            "non_duplicate_live_candidate_rows": len(non_duplicate_candidates),
            "contact_id_mismatch_rows": len(mismatch_report_rows),
            "already_written_rows": len(already_written_rows),
            "refresh_diff_rows": len([row for row in refresh_diff_rows if row.get("decision") == "refresh_candidate"]),
            "refresh_candidate_rows": len(refresh_candidates),
            "readback_missing_rows": len(readback_missing_rows),
        },
        "risk_counts": dict(Counter(row.get("decision", "") for row in [*text_report_rows, *mismatch_report_rows, *refresh_diff_rows])),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "policy": {
            "read_only": True,
            "write_crm": False,
            "write_amo": False,
            "live_write_executed": False,
            "manual_duplicate_intake_required": False,
            "requires_quality_gate_before_dry_run": True,
            "requires_real_tunnel_dry_run_before_live": True,
            "requires_readback_before_refresh": True,
            "fail_closed": True,
        },
        "next_actions": _next_actions(len(non_duplicate_candidates), len(refresh_candidates), len(readback_missing_rows), len(mismatch_report_rows)),
    }
    _write_json(outputs["summary_json"], summary)
    outputs["dashboard_html"].write_text(_render_dashboard(summary), encoding="utf-8")
    outputs["readme_md"].write_text(_render_readme(summary), encoding="utf-8")
    return summary


def _evaluate_text_quality_rows(rows: list[dict[str, str]], *, analysis_date: str) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    report: list[dict[str, str]] = []
    candidates: list[dict[str, str]] = []
    refresh: list[dict[str, str]] = []
    for row in rows:
        payload = _build_contact_payload(row)
        text = " ".join(_safe_text(row.get(field)) for field in CRM_TEXT_FIELDS)
        writeback_findings = detect_crm_writeback_quality_risks(text, min_severity="P2")
        row_text_findings = detect_crm_text_quality_risks(row, min_severity="P2", analysis_date=analysis_date)
        payload_findings = detect_crm_text_quality_risks(payload, min_severity="P2", analysis_date=analysis_date)
        blocking = bool(writeback_findings) or has_blocking_crm_text_findings(row_text_findings) or has_blocking_crm_text_findings(payload_findings)
        written = _safe_text(row.get("written_status")).casefold() == "written"
        source_ids = _split_ids(row.get("source_amo_contact_ids") or row.get("AMO contact IDs"))
        decision = "text_quality_still_blocked" if blocking else ("text_quality_cleared_already_written" if written else "text_quality_cleared")
        report_row = {
            "source_row_index": _safe_text(row.get("source_row_index")),
            "phone": _phone(row),
            "source_amo_contact_ids": " | ".join(source_ids),
            "written_status": _safe_text(row.get("written_status")),
            "written_contact_id": _safe_text(row.get("written_contact_id")),
            "queue_reason": _safe_text(row.get("queue_reason")),
            "decision": decision,
            "blocking": str(blocking).lower(),
            "risk_types": " | ".join(sorted({f.risk_type for f in [*writeback_findings, *row_text_findings, *payload_findings]})),
            "next_action": "refresh_candidate_after_readback" if written and not blocking else ("quality_gate_and_dry_run" if not blocking else "keep_blocked"),
        }
        report.append(report_row)
        if blocking:
            continue
        candidate = _candidate_row(row, source_ids[0] if len(source_ids) == 1 else "")
        if written:
            refresh.append(candidate)
        elif len(source_ids) == 1:
            candidates.append(candidate)
        else:
            report_row["decision"] = "text_quality_cleared_but_contact_ambiguous"
            report_row["next_action"] = "keep_blocked_until_single_contact"
    return report, candidates, refresh


def _evaluate_mismatch_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    report: list[dict[str, str]] = []
    recheck: list[dict[str, str]] = []
    for row in rows:
        source_ids = _split_ids(row.get("source_amo_contact_ids") or row.get("AMO contact IDs"))
        dry_id = _safe_text(row.get("dry_run_contact_id"))
        report.append(
            {
                "source_row_index": _safe_text(row.get("source_row_index")),
                "phone": _phone(row),
                "source_amo_contact_ids": " | ".join(source_ids),
                "dry_run_contact_id": dry_id,
                "queue_reason": _safe_text(row.get("queue_reason")),
                "decision": "contact_id_mismatch_stays_blocked",
                "blocking": "true",
                "risk_types": "contact_id_mismatch",
                "next_action": "operator_verify_or_merge_then_recheck",
            }
        )
        recheck.append(_candidate_row(row, source_ids[0] if len(source_ids) == 1 else ""))
    return report, recheck


def _build_refresh_diff(
    *,
    already_written_rows: list[dict[str, str]],
    extra_written_rows: list[dict[str, str]],
    readback_by_contact_id: Mapping[str, tuple[Path, Mapping[str, str]]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    report: list[dict[str, str]] = []
    candidates: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for row in [*already_written_rows, *extra_written_rows]:
        contact_id = _safe_text(row.get("written_contact_id")) or _safe_text(row.get("effective_contact_id")) or _first_id(row.get("AMO contact IDs"))
        payload = _build_contact_payload(row)
        readback = readback_by_contact_id.get(contact_id)
        base = {
            "source_row_index": _safe_text(row.get("source_row_index")),
            "phone": _phone(row),
            "contact_id": contact_id,
            "written_report": _safe_text(row.get("written_report")),
            "decision": "",
            "changed_fields": "",
            "readback_report": "",
            "next_action": "",
        }
        if not readback:
            missing.append({**base, "decision": "readback_missing", "next_action": "run_readback_before_refresh"})
            report.append({**base, "decision": "readback_missing", "next_action": "run_readback_before_refresh"})
            continue
        readback_path, readback_row = readback
        if _safe_text(readback_row.get("decision")) == "block":
            report.append(
                {
                    **base,
                    "decision": "readback_blocked",
                    "changed_fields": _safe_text(readback_row.get("risk_types")),
                    "readback_report": str(readback_path),
                    "next_action": "fix_readback_quality_before_refresh",
                }
            )
            continue
        changed = [
            field
            for field in TARGET_CONTACT_FIELDS
            if _safe_text(readback_row.get(f"field::{field}")) != _safe_text(payload.get(field))
        ]
        if changed:
            report.append(
                {
                    **base,
                    "decision": "refresh_candidate",
                    "changed_fields": " | ".join(changed),
                    "readback_report": str(readback_path),
                    "next_action": "quality_gate_and_dry_run_refresh",
                }
            )
            candidates.append(_candidate_row(row, contact_id, reason="already_written_refresh_payload_diff"))
        else:
            report.append({**base, "decision": "up_to_date", "readback_report": str(readback_path), "next_action": "none"})
    return report, candidates, missing


def _latest_readback_by_contact_id(root: Path) -> dict[str, tuple[Path, Mapping[str, str]]]:
    result: dict[str, tuple[Path, Mapping[str, str]]] = {}
    for path in sorted(root.glob("*/readback_gate*/readback_report.csv")):
        for row in _read_csv(path):
            contact_id = _safe_text(row.get("contact_id"))
            if contact_id:
                result[contact_id] = (path, row)
    return result


def _missing_readback_report_rows(rows: list[Mapping[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        result.append(
            {
                "row_index": str(index),
                "mode": "live_write",
                "phone": _safe_text(row.get("phone")),
                "status": "written",
                "reason": "",
                "contact_id": _safe_text(row.get("contact_id")),
                "contact_name": "",
                "updated_fields": json.dumps(list(TARGET_CONTACT_FIELDS), ensure_ascii=False),
                "preview_payload": "{}",
            }
        )
    return result


def _candidate_row(row: Mapping[str, str], contact_id: str, *, reason: str = "text_quality_cleared") -> dict[str, str]:
    candidate = dict(row)
    if contact_id:
        candidate["AMO contact IDs"] = contact_id
        candidate["effective_contact_id"] = contact_id
    candidate["CRM writeback policy"] = "live_update_ready"
    candidate["CRM writeback blockers"] = ""
    candidate["AMO entity policy"] = "update_existing_single_amo_contact"
    candidate["Готово к записи в AMO"] = "Да"
    candidate["Причина статуса AMO"] = reason
    return candidate


def _write_quality_and_dry_run_commands(
    *,
    project_root: Path,
    candidate_csv: Path,
    quality_out: Path,
    quality_script: Path,
    dry_run_script: Path,
    stage15_summary: Path,
    frozen_corpus_jsonl: Path,
    tenant_config_path: Optional[Path],
    analysis_date: str,
    rows: int,
) -> None:
    if rows <= 0:
        quality_body = "#!/usr/bin/env bash\nset -euo pipefail\necho \"No rows for CRM quality gate.\"\nexit 0\n"
        dry_body = "#!/usr/bin/env bash\nset -euo pipefail\necho \"No rows for AMO dry-run.\"\nexit 0\n"
    else:
        tenant_arg = f' \\\n  --tenant-config "{tenant_config_path}"' if tenant_config_path else ""
        quality_body = f'''#!/usr/bin/env bash
set -euo pipefail
cd "{project_root}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/run_crm_writeback_quality_gate.py \
  --input "{candidate_csv}" \
  --out-root "{quality_out}" \
  --frozen-corpus-jsonl "{frozen_corpus_jsonl}" \
  --population-recall-mode fail-live \
  --population-high-precision-uncovered-max 0 \
  --analysis-date "{analysis_date}"{tenant_arg}
'''
        dry_body = f'''#!/usr/bin/env bash
set -euo pipefail
cd "{project_root}"
if [ ! -f "{quality_out / 'summary.json'}" ]; then
  echo "Missing CRM quality summary: {quality_out / 'summary.json'}" >&2
  exit 2
fi
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/private/tmp/uv-cache uv run \
  --with pandas --with openpyxl --with xlsxwriter \
  --with sqlalchemy --with requests --with 'psycopg[binary]' \
  python scripts/write_amo_ready_contacts.py \
  --input "{candidate_csv}" \
  --expected-dry-run {rows} \
  --quality-gate-summary "{stage15_summary}" \
  --crm-writeback-quality-summary "{quality_out / 'summary.json'}"
'''
    _write_executable(quality_script, quality_body)
    _write_executable(dry_run_script, dry_body)


def _write_readback_missing_command(path: Path, *, project_root: Path, report_csv: Path, out_root: Path, rows: int) -> None:
    if rows <= 0:
        body = "#!/usr/bin/env bash\nset -euo pipefail\necho \"No missing readback rows.\"\nexit 0\n"
    else:
        body = f'''#!/usr/bin/env bash
set -euo pipefail
cd "{project_root}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/readback_amo_contact_writeback.py \
  --writeback-report "{report_csv}" \
  --out-root "{out_root}" \
  --expected-evaluated {rows}
'''
    _write_executable(path, body)


def _write_post_merge_full_command(path: Path, *, project_root: Path) -> None:
    body = f'''#!/usr/bin/env bash
set -euo pipefail
cd "{project_root}"
stable_runtime/amo_duplicate_resolution_20260511_v1/next_recheck_command.sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/run_amo_duplicate_after_staff_done.py --project-root . --analysis-date "{date.today().isoformat()}"
AFTER_ROOT="stable_runtime/amo_duplicate_after_staff_done_20260511_v1"
if awk 'NR > 1 {{ found=1; exit }} END {{ exit found ? 0 : 1 }}' "$AFTER_ROOT/post_merge_live_candidates_ru.csv"; then
  "$AFTER_ROOT/next_quality_gate_command.sh"
  "$AFTER_ROOT/next_real_tunnel_dry_run_command.sh"
fi
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/mango_office_operator_status.py \
  --project-root . \
  --out-root stable_runtime/operator_status_20260511_after_post_merge_full
'''
    _write_executable(path, body)


def _next_actions(text_candidates: int, refresh_candidates: int, readback_missing: int, mismatch: int) -> list[Mapping[str, Any]]:
    actions: list[Mapping[str, Any]] = []
    if text_candidates:
        actions.append({"action": "run_non_duplicate_quality_gate_and_dry_run", "rows": text_candidates})
    if refresh_candidates:
        actions.append({"action": "run_refresh_quality_gate_and_dry_run", "rows": refresh_candidates})
    if readback_missing:
        actions.append({"action": "run_readback_for_missing_written_rows", "rows": readback_missing})
    if mismatch:
        actions.append({"action": "keep_contact_id_mismatch_blocked_until_operator_verifies", "rows": mismatch})
    return actions or [{"action": "wait_for_duplicate_staff_done", "rows": 0}]


def _status(*, text_candidates: int, refresh_candidates: int, readback_missing: int, mismatch: int) -> str:
    if text_candidates or refresh_candidates:
        return "prepared_safe_next_batches"
    if readback_missing or mismatch:
        return "prepared_checks_but_no_candidates"
    return "waiting_for_duplicate_staff_done"


def _unique_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    for row in rows:
        key = (_phone(row), _safe_text(row.get("AMO contact IDs")))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _phone(row: Mapping[str, Any]) -> str:
    return normalize_phone(row.get("normalized_phone") or row.get("Телефон клиента") or row.get("phone")) or _safe_text(
        row.get("normalized_phone") or row.get("Телефон клиента") or row.get("phone")
    )


def _split_ids(value: Any) -> list[str]:
    return [part.strip() for part in _safe_text(value).replace(",", "|").split("|") if part.strip()]


def _first_id(value: Any) -> str:
    ids = _split_ids(value)
    return ids[0] if ids else ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _read_headers(path: Optional[Path]) -> list[str]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh).fieldnames or [])


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: Optional[list[str]] = None) -> None:
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
        writer.writerows(rows)


def _read_json(path: Optional[Path]) -> Mapping[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _resolve(project_root: Path, path: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def _path_from_text(value: Any) -> Optional[Path]:
    text = _safe_text(value)
    return Path(text).expanduser().resolve(strict=False) if text else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _render_readme(summary: Mapping[str, Any]) -> str:
    counts = _mapping(summary.get("counts"))
    return f"""# AMO waiting autonomous work

This folder contains safe work prepared while employees merge duplicate contacts.

- Status: `{summary.get('status')}`
- Non-duplicate live candidates: `{counts.get('non_duplicate_live_candidate_rows')}`
- Refresh candidates for already written rows: `{counts.get('refresh_candidate_rows')}`
- Missing readback rows: `{counts.get('readback_missing_rows')}`
- Contact-id mismatch rows still blocked: `{counts.get('contact_id_mismatch_rows')}`

No live AMO write is authorized here. Use generated quality-gate and dry-run commands first.
"""


def _render_dashboard(summary: Mapping[str, Any]) -> str:
    counts = _mapping(summary.get("counts"))
    cards = "\n".join(
        f"<div class='card'><span>{key}</span><strong>{value}</strong></div>"
        for key, value in counts.items()
    )
    actions = "\n".join(
        f"<li><code>{action.get('action')}</code>: {action.get('rows')} rows</li>"
        for action in summary.get("next_actions") or []
    )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>AMO waiting work</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f7f2e8;color:#16202a;margin:0}}
main{{max-width:1180px;margin:0 auto;padding:32px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.card{{background:#fffdf7;border:1px solid #dfd5c4;border-radius:16px;padding:16px;box-shadow:0 12px 26px rgba(0,0,0,.06)}}
.card span{{display:block;color:#667085;font-size:12px;text-transform:uppercase;font-weight:700}}
.card strong{{display:block;font-size:28px;margin-top:8px}}
section{{background:#fffdf7;border:1px solid #dfd5c4;border-radius:16px;margin-top:16px;padding:20px}}
code{{background:#eee6d8;padding:2px 5px;border-radius:6px}}
</style></head><body><main>
<h1>AMO waiting autonomous work</h1>
<p>Status: <code>{summary.get('status')}</code>. Read-only, no live write.</p>
<div class="grid">{cards}</div>
<section><h2>Next actions</h2><ul>{actions}</ul></section>
</main></body></html>"""


__all__ = [
    "AMO_WAITING_AUTONOMOUS_WORK_SCHEMA_VERSION",
    "DEFAULT_OUT_ROOT",
    "build_amo_waiting_autonomous_work",
]
