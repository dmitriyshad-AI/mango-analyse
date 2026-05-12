from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.amo_manual_resolution import build_amo_manual_resolution_pack
from mango_mvp.productization.amo_resolution_workbook import export_decisions_from_amo_resolution_workbook


AMO_RESOLUTION_PIPELINE_SCHEMA_VERSION = "amo_resolution_after_xlsx_pipeline_v1"
DEFAULT_PACK_ROOT = Path("stable_runtime/amo_manual_resolution_20260511_v1")
DEFAULT_WORKBOOK = DEFAULT_PACK_ROOT / "resolution_decisions_manual_template.xlsx"
DEFAULT_OUT_ROOT = Path("stable_runtime/amo_manual_resolution_20260511_v2_after_xlsx")
DEFAULT_OPERATOR_STATUS_ROOT = Path("stable_runtime/operator_status_20260511_v1")
DEFAULT_AUDIT_INBOX = Path("audits/_inbox/amo_manual_resolution_after_xlsx_20260511_v1")


@dataclass(frozen=True)
class AmoResolutionAfterXlsxConfig:
    project_root: Path
    pack_root: Path = DEFAULT_PACK_ROOT
    workbook_path: Path = DEFAULT_WORKBOOK
    out_root: Path = DEFAULT_OUT_ROOT
    audit_pack_root: Path = DEFAULT_AUDIT_INBOX
    run_quality_gate: bool = True
    update_operator_status: bool = True


def run_amo_resolution_after_xlsx_pipeline(config: AmoResolutionAfterXlsxConfig) -> Mapping[str, Any]:
    """Run the fail-closed post-XLSX AMO manual-resolution pipeline.

    This function never writes to AMO. It converts the filled workbook to the
    decisions CSV, builds an accepted-only candidate pack, optionally runs the
    CRM writeback quality gate, writes a dry-run-only command, and prepares a
    Claude audit pack.
    """

    project_root = config.project_root.expanduser().resolve(strict=False)
    pack_root = _resolve(project_root, config.pack_root)
    workbook_path = _resolve(project_root, config.workbook_path)
    out_root = _resolve(project_root, config.out_root)
    audit_pack_root = _resolve(project_root, config.audit_pack_root)
    out_root.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc)

    runtime = _read_json(project_root / "stable_runtime" / "CURRENT_RUNTIME.json")
    runtime_paths = _mapping(runtime.get("paths"))
    source_pack = _read_json(pack_root / "summary.json")
    source_summary = _mapping(source_pack.get("summary"))
    queue_root = _path_from_value(source_summary.get("queue_root")) or _path_from_value(runtime_paths.get("amo_queue_summary"))
    if queue_root and queue_root.name == "summary.json":
        queue_root = queue_root.parent
    source_csv = _path_from_value(source_summary.get("source_csv")) or _path_from_value(runtime_paths.get("amo_export_ready_csv"))
    if queue_root is None or source_csv is None:
        raise ValueError("Cannot resolve queue_root/source_csv from manual-resolution pack or CURRENT_RUNTIME.json.")

    decisions_csv = out_root / "resolution_decisions_from_xlsx.csv"
    conversion = export_decisions_from_amo_resolution_workbook(workbook_path=workbook_path, out_csv=decisions_csv)
    pack = build_amo_manual_resolution_pack(
        queue_root=queue_root,
        source_csv=source_csv,
        out_root=out_root,
        decisions_csv=decisions_csv,
        generated_at=generated_at,
    )
    qa = _build_decision_qa_report(out_root=out_root, pack=pack, generated_at=generated_at)
    quality = _maybe_run_quality_gate(
        project_root=project_root,
        out_root=out_root,
        candidates_csv=Path(pack["outputs"]["resolved_live_candidates_csv"]),
        runtime_paths=runtime_paths,
        run_quality_gate=config.run_quality_gate,
    )
    dry_run_command = _write_next_real_tunnel_dry_run_command(
        project_root=project_root,
        out_root=out_root,
        candidates_csv=Path(pack["outputs"]["resolved_live_candidates_csv"]),
        runtime_paths=runtime_paths,
        quality_summary_path=_path_from_value(_mapping(quality).get("summary_json")),
        quality_passed=bool(_mapping(quality).get("passed")),
    )
    operator_status = _maybe_update_operator_status(project_root=project_root, enabled=config.update_operator_status)
    summary = {
        "schema_version": AMO_RESOLUTION_PIPELINE_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "pack_root": str(pack_root),
        "workbook_path": str(workbook_path),
        "out_root": str(out_root),
        "decisions_csv": str(decisions_csv),
        "conversion": conversion,
        "manual_resolution_summary": pack.get("summary"),
        "decision_qa": _mapping(qa.get("summary")),
        "quality_gate": quality,
        "operator_status": operator_status,
        "audit_pack": {},
        "next_real_tunnel_dry_run_command": str(dry_run_command),
        "safety": {
            "write_crm": False,
            "live_write": False,
            "run_asr": False,
            "run_ra": False,
            "fail_closed": True,
        },
        "next_actions": _next_actions(pack, quality),
    }
    _write_json(out_root / "pipeline_summary.json", summary)
    audit_pack = _build_audit_pack(
        audit_pack_root=audit_pack_root,
        runtime_path=project_root / "stable_runtime" / "CURRENT_RUNTIME.json",
        out_root=out_root,
        qa_report=qa,
        quality=quality,
        operator_status=operator_status,
    )
    summary["audit_pack"] = audit_pack
    _write_json(out_root / "pipeline_summary.json", summary)
    pipeline_target = audit_pack_root / "pipeline_summary.json"
    if pipeline_target.exists():
        pipeline_target.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_pipeline_readme(out_root / "README_after_xlsx.md", summary)
    return summary


def _build_decision_qa_report(*, out_root: Path, pack: Mapping[str, Any], generated_at: datetime) -> Mapping[str, Any]:
    applied = _read_csv(out_root / "resolution_decisions_applied.csv")
    candidates = _read_csv(out_root / "resolved_live_candidates_ru.csv")
    still_blocked = _read_csv(out_root / "still_blocked.csv")
    needs_human = _read_csv(out_root / "needs_human.csv")
    already_written = _read_csv(out_root / "already_written_review.csv")
    accepted = [row for row in applied if _safe_text(row.get("resolution_status")).startswith("accepted")]
    risky: list[dict[str, Any]] = []
    for row in accepted:
        reason = _safe_text(row.get("resolution_reason"))
        risk_types: list[str] = []
        if _safe_text(row.get("allow_contact_id_outside_source")).casefold() in {"yes", "true", "1", "да"}:
            risk_types.append("outside_source_contact_id")
        if _safe_text(row.get("queue_bucket")) == "needs_text_quality_review":
            risk_types.append("text_quality_review_acceptance")
        if _safe_text(row.get("written_status")).casefold() == "written":
            risk_types.append("already_written_refresh")
        if not reason:
            risk_types.append("missing_resolution_reason")
        if not _safe_text(row.get("resolved_by")):
            risk_types.append("missing_resolved_by")
        if risk_types:
            risky.append(
                {
                    "resolution_id": row.get("resolution_id", ""),
                    "phone": row.get("phone", ""),
                    "risk_types": " | ".join(risk_types),
                    "resolution_status": row.get("resolution_status", ""),
                    "resolved_contact_id": row.get("resolved_contact_id", ""),
                    "resolution_reason": row.get("resolution_reason", ""),
                    "resolved_by": row.get("resolved_by", ""),
                }
            )
    validation_errors = [
        {
            "resolution_id": row.get("resolution_id", ""),
            "phone": row.get("phone", ""),
            "validation_error": row.get("validation_error", ""),
            "resolution_status": row.get("resolution_status", ""),
            "resolved_contact_id": row.get("resolved_contact_id", ""),
        }
        for row in still_blocked
        if row.get("validation_error")
    ]
    status_counts = Counter(_safe_text(row.get("resolution_status")) for row in applied)
    bucket_counts = Counter(_safe_text(row.get("queue_bucket")) for row in applied)
    summary = {
        "schema_version": "amo_resolution_decision_qa_v1",
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "review_rows": len(applied),
        "accepted_rows": len(accepted),
        "resolved_live_candidate_rows": len(candidates),
        "needs_human_rows": len(needs_human),
        "already_written_review_rows": len(already_written),
        "still_blocked_rows": len(still_blocked),
        "validation_error_rows": len(validation_errors),
        "risky_accepted_rows": len(risky),
        "preflight_passed": len(validation_errors) == 0,
        "ready_for_quality_gate": len(candidates) > 0 and len(validation_errors) == 0,
        "status_counts": dict(status_counts),
        "bucket_counts": dict(bucket_counts),
    }
    outputs = {
        "decision_qa_summary_json": out_root / "decision_qa_summary.json",
        "decision_qa_report_md": out_root / "decision_qa_report.md",
        "decision_validation_errors_csv": out_root / "decision_validation_errors.csv",
        "decision_risky_accepted_rows_csv": out_root / "decision_risky_accepted_rows.csv",
    }
    _write_json(outputs["decision_qa_summary_json"], {"summary": summary, "outputs": {k: str(v) for k, v in outputs.items()}})
    _write_csv(outputs["decision_validation_errors_csv"], validation_errors)
    _write_csv(outputs["decision_risky_accepted_rows_csv"], risky)
    _write_decision_qa_markdown(outputs["decision_qa_report_md"], summary, validation_errors, risky)
    return {"summary": summary, "outputs": {key: str(path) for key, path in outputs.items()}}


def _maybe_run_quality_gate(
    *,
    project_root: Path,
    out_root: Path,
    candidates_csv: Path,
    runtime_paths: Mapping[str, Any],
    run_quality_gate: bool,
) -> Mapping[str, Any]:
    rows = _read_csv(candidates_csv)
    quality_root = out_root / "crm_quality_gate"
    if not rows:
        return {
            "status": "skipped_no_resolved_candidates",
            "passed": False,
            "rows": 0,
            "summary_json": "",
        }
    if not run_quality_gate:
        return {
            "status": "skipped_by_config",
            "passed": False,
            "rows": len(rows),
            "summary_json": "",
        }
    source_quality_summary = _read_json(_path_from_value(runtime_paths.get("crm_quality_summary")))
    tenant_config_path = _safe_text(_mapping(source_quality_summary.get("tenant_config")).get("path"))
    frozen_corpus_jsonl = _safe_text(_mapping(source_quality_summary.get("frozen_corpus")).get("corpus_jsonl"))
    if not frozen_corpus_jsonl:
        frozen_corpus_jsonl = str(project_root / "tests" / "fixtures" / "crm_writeback_relevance_frozen_corpus.jsonl")
    cmd = [
        sys.executable,
        "scripts/run_crm_writeback_quality_gate.py",
        "--input",
        str(candidates_csv),
        "--out-root",
        str(quality_root),
        "--frozen-corpus-jsonl",
        frozen_corpus_jsonl,
        "--population-recall-mode",
        "fail-live",
        "--population-high-precision-uncovered-max",
        "0",
    ]
    if tenant_config_path:
        cmd.extend(["--tenant-config", tenant_config_path])
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:."
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(cmd, cwd=project_root, env=env, text=True, capture_output=True, check=False)
    (out_root / "crm_quality_gate_command.json").write_text(
        json.dumps(
            {
                "cmd": cmd,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = quality_root / "summary.json"
    summary = _read_json(summary_path)
    return {
        "status": "ran",
        "passed": bool(summary.get("passed")) and completed.returncode == 0,
        "returncode": completed.returncode,
        "rows": len(rows),
        "summary_json": str(summary_path),
        "blocking_rows": int(summary.get("blocking_rows") or 0) if summary else None,
    }


def _write_next_real_tunnel_dry_run_command(
    *,
    project_root: Path,
    out_root: Path,
    candidates_csv: Path,
    runtime_paths: Mapping[str, Any],
    quality_summary_path: Optional[Path],
    quality_passed: bool,
) -> Path:
    command_path = out_root / "next_real_tunnel_dry_run_command.sh"
    stage15 = _path_from_value(runtime_paths.get("stage15_summary"))
    rows = _read_csv(candidates_csv)
    if not rows:
        body = """#!/usr/bin/env bash
set -euo pipefail
echo "No resolved live candidates. Real-tunnel dry-run is intentionally skipped."
exit 0
"""
    elif not quality_passed or quality_summary_path is None:
        body = """#!/usr/bin/env bash
set -euo pipefail
echo "CRM quality gate has not passed for resolved candidates. Real-tunnel dry-run is blocked."
exit 1
"""
    else:
        body = f'''#!/usr/bin/env bash
set -euo pipefail
cd "{project_root}"
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/private/tmp/uv-cache uv run \
  --with pandas --with openpyxl --with xlsxwriter \
  --with sqlalchemy --with requests --with 'psycopg[binary]' \
  python scripts/write_amo_ready_contacts.py \
  --input "{candidates_csv}" \
  --quality-gate-summary "{stage15}" \
  --crm-writeback-quality-summary "{quality_summary_path}"
'''
    command_path.write_text(body, encoding="utf-8")
    command_path.chmod(0o755)
    return command_path


def _maybe_update_operator_status(*, project_root: Path, enabled: bool) -> Mapping[str, Any]:
    if not enabled:
        return {"status": "skipped_by_config"}
    cmd = [
        sys.executable,
        "scripts/mango_office_operator_status.py",
        "--out-root",
        str(DEFAULT_OPERATOR_STATUS_ROOT),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:."
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(cmd, cwd=project_root, env=env, text=True, capture_output=True, check=False)
    return {
        "status": "ran",
        "returncode": completed.returncode,
        "summary_json": str(project_root / DEFAULT_OPERATOR_STATUS_ROOT / "operator_status.json"),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _build_audit_pack(
    *,
    audit_pack_root: Path,
    runtime_path: Path,
    out_root: Path,
    qa_report: Mapping[str, Any],
    quality: Mapping[str, Any],
    operator_status: Mapping[str, Any],
) -> Mapping[str, Any]:
    audit_pack_root.mkdir(parents=True, exist_ok=True)
    files = [
        (runtime_path, "CURRENT_RUNTIME.json"),
        (out_root / "summary.json", "manual_resolution_summary.json"),
        (out_root / "pipeline_summary.json", "pipeline_summary.json"),
        (out_root / "decision_qa_summary.json", "decision_qa_summary.json"),
        (out_root / "decision_qa_report.md", "decision_qa_report.md"),
        (out_root / "resolution_decisions_from_xlsx.csv", "resolution_decisions_from_xlsx.csv"),
        (out_root / "resolution_decisions_applied.csv", "resolution_decisions_applied.csv"),
        (out_root / "resolved_live_candidates_ru.csv", "resolved_live_candidates_ru.csv"),
        (out_root / "still_blocked.csv", "still_blocked.csv"),
        (out_root / "needs_human.csv", "needs_human.csv"),
        (out_root / "already_written_review.csv", "already_written_review.csv"),
        (out_root / "next_real_tunnel_dry_run_command.sh", "next_real_tunnel_dry_run_command.sh"),
        (Path(str(operator_status.get("summary_json") or "")), "operator_status.json"),
    ]
    quality_summary = _path_from_value(quality.get("summary_json"))
    if quality_summary and quality_summary.exists():
        files.append((quality_summary, "crm_quality_gate_summary.json"))
    copied: list[str] = []
    for source, target_name in files:
        if not source or not source.exists() or not source.is_file():
            continue
        target = audit_pack_root / target_name
        target.write_bytes(source.read_bytes())
        copied.append(target_name)
    readme = _render_audit_readme(qa_report, quality)
    (audit_pack_root / "README.md").write_text(readme, encoding="utf-8")
    copied.append("README.md")
    return {
        "path": str(audit_pack_root),
        "files": copied,
    }


def _next_actions(pack: Mapping[str, Any], quality: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    summary = _mapping(pack.get("summary"))
    candidates = int(summary.get("resolved_live_candidate_rows") or 0)
    needs_human = int(summary.get("needs_human_rows") or 0)
    actions: list[Mapping[str, Any]] = []
    if needs_human:
        actions.append({"action": "finish_xlsx_manual_resolution", "rows": needs_human})
    if candidates and not bool(quality.get("passed")):
        actions.append({"action": "fix_quality_gate_before_dry_run", "rows": candidates})
    if candidates and bool(quality.get("passed")):
        actions.append({"action": "run_real_tunnel_dry_run", "rows": candidates})
    if not actions:
        actions.append({"action": "no_resolved_candidates_wait_for_xlsx_decisions", "rows": 0})
    return actions


def _write_decision_qa_markdown(path: Path, summary: Mapping[str, Any], errors: list[Mapping[str, Any]], risky: list[Mapping[str, Any]]) -> None:
    lines = [
        "# AMO decision QA report",
        "",
        f"- Review rows: `{summary.get('review_rows')}`",
        f"- Accepted rows: `{summary.get('accepted_rows')}`",
        f"- Resolved live candidates: `{summary.get('resolved_live_candidate_rows')}`",
        f"- Needs human: `{summary.get('needs_human_rows')}`",
        f"- Already-written review: `{summary.get('already_written_review_rows')}`",
        f"- Validation error rows: `{summary.get('validation_error_rows')}`",
        f"- Risky accepted rows: `{summary.get('risky_accepted_rows')}`",
        f"- Ready for quality gate: `{summary.get('ready_for_quality_gate')}`",
        "",
        "## Validation errors",
        "",
    ]
    if errors:
        lines.extend(f"- `{row.get('resolution_id')}`: {row.get('validation_error')}" for row in errors)
    else:
        lines.append("- none")
    lines.extend(["", "## Risky accepted rows", ""])
    if risky:
        lines.extend(f"- `{row.get('resolution_id')}`: {row.get('risk_types')}" for row in risky)
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pipeline_readme(path: Path, summary: Mapping[str, Any]) -> None:
    manual = _mapping(summary.get("manual_resolution_summary"))
    quality = _mapping(summary.get("quality_gate"))
    lines = [
        "# AMO resolution after XLSX pipeline",
        "",
        "This folder was generated by the fail-closed post-XLSX pipeline.",
        "",
        f"- Accepted rows: `{manual.get('accepted_rows')}`",
        f"- Resolved live candidates: `{manual.get('resolved_live_candidate_rows')}`",
        f"- Needs human: `{manual.get('needs_human_rows')}`",
        f"- Quality gate status: `{quality.get('status')}`",
        f"- Quality gate passed: `{quality.get('passed')}`",
        "",
        "No live AMO writeback is executed by this pipeline.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_audit_readme(qa_report: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    qa = _mapping(qa_report.get("summary"))
    return f"""# AMO resolution after XLSX audit pack

Scope: read-only audit of accepted decisions produced from the operator-filled XLSX.

Check:

1. Accepted decisions are explicit and validated.
2. `resolved_live_candidates_ru.csv` contains only accepted rows.
3. CRM quality gate passed before real-tunnel dry-run is allowed.
4. `next_real_tunnel_dry_run_command.sh` has no live-write flags.

Current counts:

- Accepted rows: `{qa.get('accepted_rows')}`
- Resolved live candidates: `{qa.get('resolved_live_candidate_rows')}`
- Validation errors: `{qa.get('validation_error_rows')}`
- Quality gate status: `{quality.get('status')}`
- Quality gate passed: `{quality.get('passed')}`

This pack does not authorize live AMO writeback.
"""


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _read_json(path: Optional[Path]) -> Mapping[str, Any]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve(project_root: Path, path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _path_from_value(value: Any) -> Optional[Path]:
    text = _safe_text(value)
    return Path(text).expanduser().resolve(strict=False) if text else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


__all__ = [
    "AMO_RESOLUTION_PIPELINE_SCHEMA_VERSION",
    "AmoResolutionAfterXlsxConfig",
    "run_amo_resolution_after_xlsx_pipeline",
]
