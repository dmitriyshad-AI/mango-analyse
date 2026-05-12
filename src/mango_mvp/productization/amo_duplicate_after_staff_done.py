from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.amo_duplicate_recheck import (
    DEFAULT_DUPLICATE_PACK_ROOT,
    DEFAULT_REPORTS_ROOT,
    build_amo_duplicate_post_merge_recheck,
)
from mango_mvp.utils.phone import normalize_phone


AMO_DUPLICATE_AFTER_STAFF_DONE_SCHEMA_VERSION = "amo_duplicate_after_staff_done_pipeline_v1"
DEFAULT_OUT_ROOT = Path("stable_runtime/amo_duplicate_after_staff_done_20260511_v1")
DEFAULT_CURRENT_RUNTIME_PATH = Path("stable_runtime/CURRENT_RUNTIME.json")
DEFAULT_FROZEN_CORPUS = Path("tests/fixtures/crm_writeback_relevance_frozen_corpus.jsonl")


def build_amo_duplicate_after_staff_done_pipeline(
    *,
    project_root: Path,
    duplicate_pack_root: Path = DEFAULT_DUPLICATE_PACK_ROOT,
    report_dir: Optional[Path] = None,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    current_runtime_path: Path = DEFAULT_CURRENT_RUNTIME_PATH,
    frozen_corpus_jsonl: Path = DEFAULT_FROZEN_CORPUS,
    analysis_date: Optional[str] = None,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    """Build the post-staff duplicate cleanup pipeline without manual intake.

    Employees clean AMO/Tallanto themselves. This pipeline only checks the AMO
    state after that cleanup and prepares a bounded candidate CSV for the next
    CRM quality gate/dry-run. It never writes to AMO.
    """

    project_root = project_root.expanduser().resolve(strict=False)
    duplicate_pack_root = _resolve(project_root, duplicate_pack_root)
    reports_root = _resolve(project_root, reports_root)
    out_root = _resolve(project_root, out_root)
    current_runtime_path = _resolve(project_root, current_runtime_path)
    frozen_corpus_jsonl = _resolve(project_root, frozen_corpus_jsonl)
    now = generated_at or datetime.now(timezone.utc)
    analysis_date = analysis_date or date.today().isoformat()
    out_root.mkdir(parents=True, exist_ok=True)

    runtime = _read_json(current_runtime_path)
    paths = _mapping(runtime.get("paths"))
    active_export_csv = _path_from_text(paths.get("amo_export_ready_csv"))
    stage15_summary = _path_from_text(paths.get("stage15_summary"))
    active_export_summary = _read_json(_path_from_text(paths.get("active_export_summary")))
    tenant_config_path = _path_from_text(_mapping(active_export_summary.get("tenant_config")).get("path"))

    recheck_out = out_root / "recheck_gate"
    recheck_summary = build_amo_duplicate_post_merge_recheck(
        duplicate_pack_root=duplicate_pack_root,
        report_dir=report_dir,
        reports_root=reports_root,
        out_root=recheck_out,
        generated_at=now,
    )
    row_results = _read_csv(Path(_mapping(recheck_summary.get("outputs")).get("row_results_csv", "")))
    active_rows, active_headers = _read_csv_with_headers(active_export_csv)
    active_by_phone = _rows_by_phone(active_rows)

    candidate_rows: list[dict[str, str]] = []
    blocked_rows: list[dict[str, str]] = []
    for row in row_results:
        phone = normalize_phone(row.get("phone")) or _safe_text(row.get("phone"))
        if row.get("decision") != "ready_after_merge":
            blocked_rows.append({**row, "pipeline_blocking_reason": row.get("blocking_reason") or "not_ready_after_merge"})
            continue
        source = active_by_phone.get(phone)
        if not source:
            blocked_rows.append({**row, "pipeline_blocking_reason": "phone_missing_from_active_strict_export"})
            continue
        surviving = _safe_text(row.get("surviving_contact_id"))
        if not surviving:
            blocked_rows.append({**row, "pipeline_blocking_reason": "missing_surviving_contact_id"})
            continue
        candidate = dict(source)
        candidate["AMO contact IDs"] = surviving
        candidate["CRM writeback policy"] = "live_update_ready"
        candidate["CRM writeback blockers"] = ""
        candidate["AMO entity policy"] = "update_existing_single_amo_contact"
        candidate["Готово к записи в AMO"] = "Да"
        candidate["Причина статуса AMO"] = "post-merge AMO recheck passed: one known contact remains"
        candidate_rows.append(candidate)

    outputs = {
        "candidate_csv": out_root / "post_merge_live_candidates_ru.csv",
        "blocked_csv": out_root / "post_merge_blocked_after_recheck.csv",
        "summary_json": out_root / "summary.json",
        "readme_md": out_root / "README.md",
        "next_quality_gate_command_sh": out_root / "next_quality_gate_command.sh",
        "next_real_tunnel_dry_run_command_sh": out_root / "next_real_tunnel_dry_run_command.sh",
    }
    _write_csv(outputs["candidate_csv"], candidate_rows, active_headers)
    _write_csv(outputs["blocked_csv"], blocked_rows, _merged_headers(row_results, ["pipeline_blocking_reason"]))
    _write_quality_command(
        outputs["next_quality_gate_command_sh"],
        project_root=project_root,
        input_csv=outputs["candidate_csv"],
        out_root=out_root / "crm_quality_gate",
        frozen_corpus_jsonl=frozen_corpus_jsonl,
        tenant_config_path=tenant_config_path,
        analysis_date=analysis_date,
        rows=len(candidate_rows),
    )
    _write_dry_run_command(
        outputs["next_real_tunnel_dry_run_command_sh"],
        project_root=project_root,
        input_csv=outputs["candidate_csv"],
        stage15_summary=stage15_summary,
        quality_summary=out_root / "crm_quality_gate" / "summary.json",
        rows=len(candidate_rows),
    )

    status = _pipeline_status(recheck_summary, len(candidate_rows), len(blocked_rows))
    summary = {
        "schema_version": AMO_DUPLICATE_AFTER_STAFF_DONE_SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "duplicate_pack_root": str(duplicate_pack_root),
        "out_root": str(out_root),
        "status": status,
        "active_export_csv": str(active_export_csv),
        "stage15_summary": str(stage15_summary),
        "tenant_config_path": str(tenant_config_path) if tenant_config_path else "",
        "frozen_corpus_jsonl": str(frozen_corpus_jsonl),
        "recheck_summary": recheck_summary,
        "candidate_rows": len(candidate_rows),
        "blocked_rows": len(blocked_rows),
        "candidate_contact_ids": _contact_ids(candidate_rows),
        "blocked_reason_counts": dict(Counter(row.get("pipeline_blocking_reason") or row.get("blocking_reason") or "unknown" for row in blocked_rows)),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "policy": {
            "read_only": True,
            "write_crm": False,
            "live_write_executed": False,
            "manual_intake_required": False,
            "requires_post_merge_recheck": True,
            "requires_crm_quality_gate_before_dry_run": True,
            "requires_real_tunnel_dry_run_before_live": True,
            "fail_closed": True,
        },
        "next_actions": _next_actions(status, len(candidate_rows), len(blocked_rows)),
    }
    _write_json(outputs["summary_json"], summary)
    outputs["readme_md"].write_text(_render_readme(summary), encoding="utf-8")
    return summary


def _pipeline_status(recheck_summary: Mapping[str, Any], candidate_rows: int, blocked_rows: int) -> str:
    if recheck_summary.get("status") == "pending_not_run":
        return "waiting_for_staff_done_and_recheck"
    if candidate_rows and blocked_rows:
        return "partial_ready_for_quality_gate"
    if candidate_rows:
        return "ready_for_quality_gate"
    return "blocked_no_ready_rows"


def _next_actions(status: str, candidate_rows: int, blocked_rows: int) -> list[Mapping[str, Any]]:
    if status == "waiting_for_staff_done_and_recheck":
        return [
            {
                "action": "run_post_merge_recheck_after_staff_done",
                "rows": blocked_rows,
                "description_ru": "Когда сотрудники сообщат, что дубли разобраны, запустить next_recheck_command.sh и повторить pipeline.",
            }
        ]
    actions: list[Mapping[str, Any]] = []
    if candidate_rows:
        actions.append(
            {
                "action": "run_crm_quality_gate_for_post_merge_candidates",
                "rows": candidate_rows,
                "description_ru": "Запустить next_quality_gate_command.sh, затем next_real_tunnel_dry_run_command.sh для bounded batch.",
            }
        )
    if blocked_rows:
        actions.append(
            {
                "action": "keep_failed_recheck_rows_blocked",
                "rows": blocked_rows,
                "description_ru": "Не прошедшие recheck строки остаются blocked; сотрудники/оператор исправляют только их.",
            }
        )
    return actions or [{"action": "no_rows", "rows": 0}]


def _write_quality_command(
    path: Path,
    *,
    project_root: Path,
    input_csv: Path,
    out_root: Path,
    frozen_corpus_jsonl: Path,
    tenant_config_path: Optional[Path],
    analysis_date: str,
    rows: int,
) -> None:
    if rows <= 0:
        body = """#!/usr/bin/env bash
set -euo pipefail
echo "No post-merge candidate rows for CRM quality gate."
exit 0
"""
    else:
        tenant_arg = f' \\\n  --tenant-config "{tenant_config_path}"' if tenant_config_path else ""
        body = f'''#!/usr/bin/env bash
set -euo pipefail
cd "{project_root}"
echo "Running CRM quality gate for {rows} post-merge candidate rows."
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/run_crm_writeback_quality_gate.py \
  --input "{input_csv}" \
  --out-root "{out_root}" \
  --frozen-corpus-jsonl "{frozen_corpus_jsonl}" \
  --population-recall-mode fail-live \
  --population-high-precision-uncovered-max 0 \
  --analysis-date "{analysis_date}"{tenant_arg}
'''
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _write_dry_run_command(
    path: Path,
    *,
    project_root: Path,
    input_csv: Path,
    stage15_summary: Optional[Path],
    quality_summary: Path,
    rows: int,
) -> None:
    if rows <= 0:
        body = """#!/usr/bin/env bash
set -euo pipefail
echo "No post-merge candidate rows for AMO dry-run."
exit 0
"""
    else:
        body = f'''#!/usr/bin/env bash
set -euo pipefail
cd "{project_root}"
if [ ! -f "{quality_summary}" ]; then
  echo "Missing CRM quality summary: {quality_summary}" >&2
  echo "Run next_quality_gate_command.sh first." >&2
  exit 2
fi
echo "Running real-tunnel AMO dry-run for {rows} post-merge candidate rows. No live write flags are used."
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/private/tmp/uv-cache uv run \
  --with pandas --with openpyxl --with xlsxwriter \
  --with sqlalchemy --with requests --with 'psycopg[binary]' \
  python scripts/write_amo_ready_contacts.py \
  --input "{input_csv}" \
  --expected-dry-run {rows} \
  --quality-gate-summary "{stage15_summary or ''}" \
  --crm-writeback-quality-summary "{quality_summary}"
'''
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _rows_by_phone(rows: list[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        phone = normalize_phone(row.get("Телефон клиента")) or _safe_text(row.get("Телефон клиента"))
        if phone:
            result[phone] = dict(row)
    return result


def _contact_ids(rows: list[Mapping[str, str]]) -> list[str]:
    result: list[str] = []
    for row in rows:
        contact_id = _safe_text(row.get("AMO contact IDs"))
        if contact_id and contact_id not in result:
            result.append(contact_id)
    return result


def _read_csv_with_headers(path: Optional[Path]) -> tuple[list[dict[str, str]], list[str]]:
    if not path or not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: list[Mapping[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        fieldnames = _merged_headers(rows, [])
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _merged_headers(rows: list[Mapping[str, str]], extra: list[str]) -> list[str]:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    for key in extra:
        if key not in headers:
            headers.append(key)
    return headers


def _read_json(path: Optional[Path]) -> Mapping[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    return f"""# AMO duplicate after-staff-done pipeline

This is the simplified flow: employees clean AMO/Tallanto duplicates on their own, then the system verifies AMO state and builds a bounded next batch.

- Status: `{summary.get('status')}`
- Candidate rows: `{summary.get('candidate_rows')}`
- Blocked rows: `{summary.get('blocked_rows')}`
- Manual intake required: `{_mapping(summary.get('policy')).get('manual_intake_required')}`

Current use:

1. Employees finish duplicate cleanup.
2. Run `stable_runtime/amo_duplicate_resolution_20260511_v1/next_recheck_command.sh`.
3. Run this pipeline again. It will auto-find the matching dry-run report by input path.
4. If candidates exist, run `next_quality_gate_command.sh`.
5. If quality gate is green, run `next_real_tunnel_dry_run_command.sh`.
6. Live writeback still requires separate bounded audit, approval and readback.

No AMO live write is executed by this pipeline.
"""
