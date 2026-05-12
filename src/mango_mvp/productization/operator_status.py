from __future__ import annotations

import csv
import html
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.productization.current_runtime import (
    DEFAULT_CURRENT_RUNTIME_PATH,
    build_current_runtime_contract,
    load_current_runtime_contract,
)


OPERATOR_STATUS_SCHEMA_VERSION = "operator_status_v1"
DEFAULT_OPERATOR_STATUS_ROOT = Path("stable_runtime/operator_status_20260511_v1")


QUEUE_LABELS = {
    "ready_single_contact_not_written": "Готово к staged live-write",
    "already_written": "Уже записано и подтверждено предыдущими этапами",
    "needs_manager_review_multi_contact": "Нужен ручной выбор AMO-контакта",
    "blocked_contact_id_mismatch": "Заблокировано: AMO dry-run нашел другой contact_id",
    "needs_text_quality_review": "Нужна проверка текста перед записью",
    "deferred_non_sales_or_service": "Отложено: не sales-контекст",
}

QUEUE_ACTIONS = {
    "ready_single_contact_not_written": "Можно готовить следующий dry-run/live stage, если все gates зеленые.",
    "already_written": "Действий не требуется; строку не переписывать без отдельного refresh-сценария.",
    "needs_manager_review_multi_contact": "Менеджер/оператор должен выбрать правильный AMO contact_id или объединить дубли.",
    "blocked_contact_id_mismatch": "Нужно сверить AMO-карточку и источник contact_id; автоматически писать нельзя.",
    "needs_text_quality_review": "Нужно исправить или подтвердить CRM-текст, затем пересобрать strict export/gate.",
    "deferred_non_sales_or_service": "Не писать как новый sales follow-up; использовать отдельную service/existing-client политику.",
}


@dataclass(frozen=True)
class OperatorStatusSummary:
    schema_version: str
    generated_at: str
    project_root: str
    runtime_validation_ok: bool
    call_processing_ready: bool
    crm_writeback_live_allowed_now: bool
    blocked: int
    blocked_semantics: str
    runtime_blocked: int
    production_blocking_reasons_count: int
    contact_id_mismatch_blocked_rows: int
    warnings: int
    canonical_actionable_calls: int
    canonical_missing_asr: int
    canonical_missing_ra: int
    amo_ready_rows: int
    queue_ready_rows: int
    queue_already_written_rows: int
    queue_manual_resolution_rows: int
    duplicate_merge_required_rows: int
    duplicate_contact_mismatch_rows: int
    duplicate_staff_task_rows: int
    duplicate_recheck_passed: bool
    duplicate_recheck_blocked_rows: int
    duplicate_after_staff_done_status: str
    duplicate_after_staff_done_candidate_rows: int
    duplicate_after_staff_done_blocked_rows: int
    waiting_work_status: str
    waiting_work_non_duplicate_candidate_rows: int
    waiting_work_refresh_candidate_rows: int
    waiting_work_readback_missing_rows: int
    waiting_work_contact_id_mismatch_rows: int
    waiting_work_dry_run_prepared_rows: int
    waiting_work_live_write_allowed_now: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_operator_status(
    *,
    project_root: Path,
    runtime_contract_path: Optional[Path] = None,
    out_root: Optional[Path] = None,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    """Build the read-only operator status layer for the current production runtime."""

    project_root = project_root.resolve(strict=False)
    out_root = _resolve_optional(project_root, out_root or DEFAULT_OPERATOR_STATUS_ROOT)
    now = generated_at or datetime.now(timezone.utc)
    contract_path = _resolve_optional(project_root, runtime_contract_path or DEFAULT_CURRENT_RUNTIME_PATH)
    if contract_path and contract_path.exists():
        contract = load_current_runtime_contract(contract_path)
    else:
        contract = build_current_runtime_contract(project_root=project_root, out_path=contract_path)

    paths = _mapping(contract.get("paths"))
    queue_summary_path = _path_from_value(paths.get("amo_queue_summary"))
    queue_summary = _load_json_if_exists(queue_summary_path)
    manual_resolution_summary_path = _find_latest_manual_resolution_summary(project_root, queue_summary)
    manual_resolution = _load_json_if_exists(manual_resolution_summary_path)
    duplicate_resolution_summary_path = _find_latest_duplicate_resolution_summary(project_root)
    duplicate_resolution = _load_json_if_exists(duplicate_resolution_summary_path)
    duplicate_staff_summary_path = _find_latest_summary(project_root, "amo_duplicate_staff_tasks_*")
    duplicate_staff = _load_json_if_exists(duplicate_staff_summary_path)
    duplicate_recheck_summary_path = _find_latest_summary(project_root, "amo_duplicate_post_merge_recheck_*")
    duplicate_recheck = _load_json_if_exists(duplicate_recheck_summary_path)
    duplicate_after_staff_done_summary_path = _find_latest_summary(project_root, "amo_duplicate_after_staff_done_*")
    duplicate_after_staff_done = _load_json_if_exists(duplicate_after_staff_done_summary_path)
    waiting_work_summary_path = _find_latest_summary(project_root, "amo_waiting_autonomous_work_*")
    waiting_work = _load_json_if_exists(waiting_work_summary_path)
    stage_rollout_summary_path = _find_latest_summary(project_root, "amo_stage50_stage86_preflight_blocked_*")
    stage_rollout_summary = _load_json_if_exists(stage_rollout_summary_path)
    cleanup_manifest_summary_path = _find_latest_summary(project_root, "project_cleanup_manifest_*")
    cleanup_manifest_summary = _load_json_if_exists(cleanup_manifest_summary_path)
    queue_rows = _load_queue_rows(queue_summary)
    queue_counts = _mapping(queue_summary.get("bucket_counts"))
    queue_operator_rows = [_operator_queue_row(row) for row in queue_rows]
    queue_manual_rows = sum(
        _int(queue_counts.get(bucket))
        for bucket in ("needs_manager_review_multi_contact", "blocked_contact_id_mismatch", "needs_text_quality_review")
    )

    runtime_summary = _mapping(contract.get("summary"))
    readiness_summary = _mapping(_mapping(contract.get("readiness")).get("summary"))
    production_loop = _build_production_loop(
        contract=contract,
        queue_summary=queue_summary,
        manual_resolution=manual_resolution,
        waiting_work=waiting_work,
    )
    stage_rollout = _build_stage_rollout(stage_rollout_summary_path, stage_rollout_summary)
    cleanup_status = _build_cleanup_status(cleanup_manifest_summary_path, cleanup_manifest_summary)
    runtime_blocked = int(runtime_summary.get("blocked") or 0)
    production_blocking_reasons_count = len(production_loop["blocking_reasons"])
    contact_id_mismatch_blocked_rows = _int(queue_counts.get("blocked_contact_id_mismatch"))
    duplicate_status_counts = _mapping(duplicate_resolution.get("by_duplicate_resolution_status"))
    duplicate_merge_required_rows = _int(duplicate_status_counts.get("duplicate_contacts_merge_required")) or _int(
        queue_counts.get("needs_manager_review_multi_contact")
    )
    duplicate_contact_mismatch_rows = _int(duplicate_status_counts.get("contact_id_mismatch_requires_operator")) or contact_id_mismatch_blocked_rows
    duplicate_staff_task_rows = _int(duplicate_staff.get("task_rows"))
    duplicate_recheck_passed = bool(duplicate_recheck.get("passed"))
    duplicate_recheck_blocked_rows = _int(duplicate_recheck.get("blocked_rows"))
    duplicate_after_staff_done_status = str(duplicate_after_staff_done.get("status") or "not_built")
    duplicate_after_staff_done_candidate_rows = _int(duplicate_after_staff_done.get("candidate_rows"))
    duplicate_after_staff_done_blocked_rows = _int(duplicate_after_staff_done.get("blocked_rows"))
    waiting_work_counts = _mapping(waiting_work.get("counts"))
    waiting_work_status = str(waiting_work.get("status") or "not_built")
    waiting_work_non_duplicate_candidate_rows = _int(waiting_work_counts.get("non_duplicate_live_candidate_rows"))
    waiting_work_refresh_candidate_rows = _int(waiting_work_counts.get("refresh_candidate_rows"))
    waiting_work_readback_missing_rows = _int(waiting_work_counts.get("readback_missing_rows"))
    waiting_work_contact_id_mismatch_rows = _int(waiting_work_counts.get("contact_id_mismatch_rows"))
    waiting_work_dry_run_prepared_rows = waiting_work_non_duplicate_candidate_rows + waiting_work_refresh_candidate_rows
    blocked = runtime_blocked + production_blocking_reasons_count
    warnings = int(runtime_summary.get("warnings") or 0) + len(production_loop["warnings"])
    summary = OperatorStatusSummary(
        schema_version=OPERATOR_STATUS_SCHEMA_VERSION,
        generated_at=now.isoformat(timespec="seconds"),
        project_root=str(project_root),
        runtime_validation_ok=bool(runtime_summary.get("validation_ok")),
        call_processing_ready=bool(readiness_summary.get("processing_pipeline_ready")),
        crm_writeback_live_allowed_now=bool(production_loop["live_write_allowed_now"]),
        blocked=blocked,
        blocked_semantics="runtime_blocked_plus_production_blocking_reasons",
        runtime_blocked=runtime_blocked,
        production_blocking_reasons_count=production_blocking_reasons_count,
        contact_id_mismatch_blocked_rows=contact_id_mismatch_blocked_rows,
        warnings=warnings,
        canonical_actionable_calls=_int(runtime_summary.get("canonical_actionable_calls")),
        canonical_missing_asr=_int(runtime_summary.get("canonical_missing_asr")),
        canonical_missing_ra=_int(runtime_summary.get("canonical_missing_ra")),
        amo_ready_rows=_int(runtime_summary.get("amo_ready_rows")),
        queue_ready_rows=_int(queue_counts.get("ready_single_contact_not_written")),
        queue_already_written_rows=_int(queue_counts.get("already_written")),
        queue_manual_resolution_rows=queue_manual_rows,
        duplicate_merge_required_rows=duplicate_merge_required_rows,
        duplicate_contact_mismatch_rows=duplicate_contact_mismatch_rows,
        duplicate_staff_task_rows=duplicate_staff_task_rows,
        duplicate_recheck_passed=duplicate_recheck_passed,
        duplicate_recheck_blocked_rows=duplicate_recheck_blocked_rows,
        duplicate_after_staff_done_status=duplicate_after_staff_done_status,
        duplicate_after_staff_done_candidate_rows=duplicate_after_staff_done_candidate_rows,
        duplicate_after_staff_done_blocked_rows=duplicate_after_staff_done_blocked_rows,
        waiting_work_status=waiting_work_status,
        waiting_work_non_duplicate_candidate_rows=waiting_work_non_duplicate_candidate_rows,
        waiting_work_refresh_candidate_rows=waiting_work_refresh_candidate_rows,
        waiting_work_readback_missing_rows=waiting_work_readback_missing_rows,
        waiting_work_contact_id_mismatch_rows=waiting_work_contact_id_mismatch_rows,
        waiting_work_dry_run_prepared_rows=waiting_work_dry_run_prepared_rows,
        waiting_work_live_write_allowed_now=False,
    )
    status = {
        "summary": summary.to_json_dict(),
        "runtime_contract": {
            "path": str(contract_path) if contract_path else None,
            "summary": runtime_summary,
            "paths": paths,
        },
        "call_processing": {
            "summary": readiness_summary,
            "gates": list(_mapping(contract.get("readiness")).get("gates") or []),
            "next_actions": list(_mapping(contract.get("readiness")).get("next_actions") or []),
        },
        "crm_queue": {
            "summary_path": str(queue_summary_path) if queue_summary_path else None,
            "bucket_counts": dict(queue_counts),
            "buckets": _queue_bucket_details(queue_counts),
            "operator_rows_count": len(queue_operator_rows),
        },
        "manual_resolution": {
            "summary_path": str(manual_resolution_summary_path) if manual_resolution_summary_path else None,
            "summary": _mapping(manual_resolution.get("summary")),
            "outputs": _mapping(manual_resolution.get("outputs")),
            "next_actions": list(manual_resolution.get("next_actions") or []),
        },
        "duplicate_resolution": {
            "summary_path": str(duplicate_resolution_summary_path) if duplicate_resolution_summary_path else None,
            "summary": duplicate_resolution,
            "outputs": _mapping(duplicate_resolution.get("outputs")),
            "next_actions": list(duplicate_resolution.get("next_actions") or []),
        },
        "duplicate_staff_tasks": {
            "summary_path": str(duplicate_staff_summary_path) if duplicate_staff_summary_path else None,
            "summary": duplicate_staff,
            "outputs": _mapping(duplicate_staff.get("outputs")),
            "next_actions": list(duplicate_staff.get("next_actions") or []),
        },
        "duplicate_post_merge_recheck": {
            "summary_path": str(duplicate_recheck_summary_path) if duplicate_recheck_summary_path else None,
            "summary": duplicate_recheck,
            "outputs": _mapping(duplicate_recheck.get("outputs")),
            "next_actions": list(duplicate_recheck.get("next_actions") or []),
        },
        "duplicate_after_staff_done": {
            "summary_path": str(duplicate_after_staff_done_summary_path) if duplicate_after_staff_done_summary_path else None,
            "summary": duplicate_after_staff_done,
            "outputs": _mapping(duplicate_after_staff_done.get("outputs")),
            "next_actions": list(duplicate_after_staff_done.get("next_actions") or []),
        },
        "waiting_autonomous_work": _build_waiting_autonomous_work_status(
            waiting_work_summary_path,
            waiting_work,
            runtime_validation_ok=bool(runtime_summary.get("validation_ok")),
        ),
        "stage_rollout": stage_rollout,
        "cleanup": cleanup_status,
        "leadership_snapshot": _build_leadership_snapshot(
            summary=summary.to_json_dict(),
            manual_resolution=manual_resolution,
            duplicate_resolution=duplicate_resolution,
            duplicate_staff=duplicate_staff,
            duplicate_recheck=duplicate_recheck,
            duplicate_after_staff_done=duplicate_after_staff_done,
            waiting_work=waiting_work,
            stage_rollout=stage_rollout,
            cleanup_status=cleanup_status,
        ),
        "amo_production_loop": production_loop,
        "dashboard": {
            "operator_dashboard_html": str(out_root / "operator_dashboard.html") if out_root else None,
            "operator_markdown": str(out_root / "operator_status.md") if out_root else None,
            "crm_queue_operator_csv": str(out_root / "crm_queue_operator.csv") if out_root else None,
        },
        "safety": {
            "read_only": True,
            "downloads_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_runtime_db": False,
            "write_crm": False,
            "write_tallanto": False,
        },
    }
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)
        _write_json(out_root / "operator_status.json", status)
        _write_operator_queue_csv(out_root / "crm_queue_operator.csv", queue_operator_rows)
        (out_root / "operator_status.md").write_text(render_operator_status_markdown(status), encoding="utf-8")
        (out_root / "operator_dashboard.html").write_text(render_operator_dashboard_html(status), encoding="utf-8")
    return status


def _build_production_loop(
    *,
    contract: Mapping[str, Any],
    queue_summary: Mapping[str, Any],
    manual_resolution: Mapping[str, Any],
    waiting_work: Mapping[str, Any],
) -> Mapping[str, Any]:
    runtime_summary = _mapping(contract.get("summary"))
    readiness_summary = _mapping(_mapping(contract.get("readiness")).get("summary"))
    queue_counts = _mapping(queue_summary.get("bucket_counts"))
    ready = _int(queue_counts.get("ready_single_contact_not_written"))
    already_written = _int(queue_counts.get("already_written"))
    multi = _int(queue_counts.get("needs_manager_review_multi_contact"))
    mismatch = _int(queue_counts.get("blocked_contact_id_mismatch"))
    text_review = _int(queue_counts.get("needs_text_quality_review"))
    deferred = _int(queue_counts.get("deferred_non_sales_or_service"))
    resolution_summary = _mapping(manual_resolution.get("summary"))
    resolved_candidates = _int(resolution_summary.get("resolved_live_candidate_rows"))
    waiting_counts = _mapping(waiting_work.get("counts"))
    waiting_non_duplicate = _int(waiting_counts.get("non_duplicate_live_candidate_rows"))
    waiting_refresh = _int(waiting_counts.get("refresh_candidate_rows"))
    waiting_readback_missing = _int(waiting_counts.get("readback_missing_rows"))
    waiting_contact_mismatch = _int(waiting_counts.get("contact_id_mismatch_rows"))
    waiting_dry_run_prepared = waiting_non_duplicate + waiting_refresh
    blockers: list[str] = []
    warnings: list[str] = []
    if not bool(runtime_summary.get("validation_ok")):
        blockers.append("runtime_contract_not_green")
    if not bool(readiness_summary.get("processing_pipeline_ready")):
        blockers.append("call_processing_readiness_not_green")
    if ready == 0 and resolved_candidates == 0:
        blockers.append("no_ready_single_contact_rows_for_next_live_stage")
    if multi or mismatch or text_review:
        warnings.append("manual_resolution_queue_not_empty")
    if deferred:
        warnings.append("deferred_service_or_non_sales_rows_present")
    if waiting_readback_missing:
        warnings.append("waiting_work_readback_missing_rows_present")
    if waiting_contact_mismatch:
        warnings.append("waiting_work_contact_id_mismatch_still_blocked")
    if resolved_candidates:
        stage = "manual_resolution_candidates_ready_for_quality_gate_and_dry_run"
    elif waiting_dry_run_prepared:
        stage = "waiting_work_candidates_ready_for_real_tunnel_dry_run"
    elif ready == 0 and already_written:
        stage = "stage1_complete_no_ready_rows"
    else:
        stage = "ready_for_next_dry_run_stage"
    return {
        "schema_version": "amo_production_loop_status_v1",
        "stage": stage,
        # Live AMO writes require a separate explicit operator approval artifact.
        "live_write_allowed_now": False,
        "live_write_approval_required": True,
        "dry_run_allowed_now": bool(runtime_summary.get("validation_ok")) and ready > 0,
        "resolved_dry_run_allowed_now": bool(runtime_summary.get("validation_ok")) and resolved_candidates > 0,
        "waiting_work_dry_run_allowed_now": bool(runtime_summary.get("validation_ok")) and waiting_dry_run_prepared > 0,
        "any_dry_run_allowed_now": bool(runtime_summary.get("validation_ok")) and (ready > 0 or resolved_candidates > 0 or waiting_dry_run_prepared > 0),
        "waiting_work_summary": {
            "status": waiting_work.get("status") or "not_built",
            "non_duplicate_live_candidate_rows": waiting_non_duplicate,
            "refresh_candidate_rows": waiting_refresh,
            "readback_missing_rows": waiting_readback_missing,
            "contact_id_mismatch_rows": waiting_contact_mismatch,
            "dry_run_prepared_rows": waiting_dry_run_prepared,
            "live_write_allowed_now": False,
        },
        "blocking_reasons": blockers,
        "warnings": warnings,
        "queue_counts": dict(queue_counts),
        "manual_resolution_summary": dict(resolution_summary),
        "required_sequence": [
            "strict_export",
            "crm_quality_gate",
            "queue_classification",
            "real_tunnel_dry_run",
            "explicit_operator_live_approval_artifact",
            "operator_live_confirmation",
            "staged_live_write",
            "post_writeback_readback",
            "queue_rebuild",
        ],
        "next_operator_actions": _production_next_actions(
            ready=ready,
            resolved_candidates=resolved_candidates,
            multi=multi,
            mismatch=mismatch,
            text_review=text_review,
            waiting_non_duplicate=waiting_non_duplicate,
            waiting_refresh=waiting_refresh,
            waiting_readback_missing=waiting_readback_missing,
            waiting_contact_mismatch=waiting_contact_mismatch,
        ),
    }


def _production_next_actions(
    *,
    ready: int,
    resolved_candidates: int,
    multi: int,
    mismatch: int,
    text_review: int,
    waiting_non_duplicate: int,
    waiting_refresh: int,
    waiting_readback_missing: int,
    waiting_contact_mismatch: int,
) -> list[Mapping[str, Any]]:
    actions: list[Mapping[str, Any]] = []
    if resolved_candidates:
        actions.append(
            {
                "action": "run_resolved_candidates_quality_gate_and_real_tunnel_dry_run",
                "rows": resolved_candidates,
                "owner": "operator",
                "description_ru": "Прогнать CRM quality gate и AMO dry-run по подтвержденным manual-resolution кандидатам.",
            }
        )
    if ready:
        actions.append(
            {
                "action": "prepare_next_live_stage",
                "rows": ready,
                "owner": "operator",
                "description_ru": "Подготовить staged dry-run/live/readback для готовых single-contact строк.",
            }
        )
    if multi:
        actions.append(
            {
                "action": "merge_duplicate_amo_contacts",
                "rows": multi,
                "owner": "sales_ops_or_responsible_manager",
                "description_ru": "Объединить AMO-дубли по одному телефону; до post-merge recheck эти строки нельзя писать live.",
            }
        )
    if mismatch:
        actions.append(
            {
                "action": "resolve_contact_id_mismatch",
                "rows": mismatch,
                "owner": "operator",
                "description_ru": "Сверить AMO dry-run contact_id с source AMO contact IDs; автоматическая запись заблокирована.",
            }
        )
    if text_review:
        actions.append(
            {
                "action": "review_crm_text_quality_rows",
                "rows": text_review,
                "owner": "manager_or_ai_auditor",
                "description_ru": "Проверить CRM-текст, исправить источник и пересобрать strict export/gate.",
            }
        )
    if waiting_readback_missing:
        actions.append(
            {
                "action": "run_waiting_work_readback_missing_rows",
                "rows": waiting_readback_missing,
                "owner": "operator",
                "description_ru": "Прочитать обратно уже записанные AMO-карточки; refresh по этим строкам запрещен до успешного readback.",
            }
        )
    if waiting_non_duplicate:
        actions.append(
            {
                "action": "run_waiting_work_non_duplicate_real_tunnel_dry_run",
                "rows": waiting_non_duplicate,
                "owner": "operator",
                "description_ru": "Прогнать real-tunnel dry-run по безопасным non-duplicate кандидатам; live только после отдельного approval.",
            }
        )
    if waiting_refresh:
        actions.append(
            {
                "action": "run_waiting_work_refresh_real_tunnel_dry_run",
                "rows": waiting_refresh,
                "owner": "operator",
                "description_ru": "Прогнать diff-based refresh dry-run по уже записанным контактам; не делать broad rewrite.",
            }
        )
    if waiting_contact_mismatch:
        actions.append(
            {
                "action": "keep_waiting_work_contact_id_mismatch_blocked",
                "rows": waiting_contact_mismatch,
                "owner": "operator",
                "description_ru": "Не принимать contact_id mismatch автоматически; нужна сверка карточки/источника.",
            }
        )
    if not actions:
        actions.append(
            {
                "action": "wait_for_new_post_backfill_candidates",
                "rows": 0,
                "owner": "operator",
                "description_ru": "Готовых строк для live-записи нет; следующий шаг зависит от новых кандидатов или ручного разбора блокировок.",
            }
        )
    return actions


def _queue_bucket_details(counts: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = []
    for bucket, label in QUEUE_LABELS.items():
        count = _int(counts.get(bucket))
        items.append(
            {
                "bucket": bucket,
                "label_ru": label,
                "rows": count,
                "action_ru": QUEUE_ACTIONS[bucket],
                "live_write_safe": bucket == "ready_single_contact_not_written",
            }
        )
    return items


def _build_stage_rollout(path: Optional[Path], payload: Mapping[str, Any]) -> Mapping[str, Any]:
    stage50_allowed = bool(payload.get("stage50_preflight_allowed"))
    stage86_allowed = bool(payload.get("stage86_preflight_allowed"))
    return {
        "summary_path": str(path) if path else None,
        "summary": dict(payload),
        "stage50_preflight_allowed": stage50_allowed,
        "stage86_preflight_allowed": stage86_allowed,
        "stage50_blocked": not stage50_allowed,
        "stage86_blocked": not stage86_allowed,
        "reason": payload.get("reason") or payload.get("blocked_reason") or "",
    }


def _build_cleanup_status(path: Optional[Path], payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "summary_path": str(path) if path else None,
        "summary": dict(payload),
        "candidate_rows": _int(payload.get("candidate_rows")),
        "safe_to_quarantine_rows": _int(payload.get("safe_to_quarantine_rows")),
        "requires_human_review_rows": _int(payload.get("requires_human_review_rows")),
        "read_only_scan": bool(_mapping(payload.get("safety")).get("read_only_scan")),
    }


def _build_waiting_autonomous_work_status(
    path: Optional[Path],
    payload: Mapping[str, Any],
    *,
    runtime_validation_ok: bool,
) -> Mapping[str, Any]:
    counts = _mapping(payload.get("counts"))
    outputs = _mapping(payload.get("outputs"))
    non_duplicate = _int(counts.get("non_duplicate_live_candidate_rows"))
    refresh = _int(counts.get("refresh_candidate_rows"))
    readback_missing = _int(counts.get("readback_missing_rows"))
    mismatch = _int(counts.get("contact_id_mismatch_rows"))
    dry_run_prepared = non_duplicate + refresh
    return {
        "summary_path": str(path) if path else None,
        "summary": dict(payload),
        "status": payload.get("status") or "not_built",
        "counts": {
            "non_duplicate_live_candidate_rows": non_duplicate,
            "refresh_candidate_rows": refresh,
            "readback_missing_rows": readback_missing,
            "contact_id_mismatch_rows": mismatch,
            "already_written_rows": _int(counts.get("already_written_rows")),
            "text_quality_review_rows": _int(counts.get("text_quality_review_rows")),
            "text_quality_cleared_rows": _int(counts.get("text_quality_cleared_rows")),
            "dry_run_prepared_rows": dry_run_prepared,
        },
        "outputs": outputs,
        "next_actions": list(payload.get("next_actions") or []),
        "dry_run_allowed_when_tunnel_available": runtime_validation_ok and dry_run_prepared > 0,
        "live_write_allowed_now": False,
        "policy": {
            "read_only": True,
            "requires_real_tunnel_dry_run_before_live": True,
            "requires_explicit_live_approval": True,
            "requires_readback_before_refresh": True,
            "contact_id_mismatch_stays_blocked": mismatch > 0,
            "write_crm": False,
        },
    }


def _build_leadership_snapshot(
    *,
    summary: Mapping[str, Any],
    manual_resolution: Mapping[str, Any],
    duplicate_resolution: Mapping[str, Any],
    duplicate_staff: Mapping[str, Any],
    duplicate_recheck: Mapping[str, Any],
    duplicate_after_staff_done: Mapping[str, Any],
    waiting_work: Mapping[str, Any],
    stage_rollout: Mapping[str, Any],
    cleanup_status: Mapping[str, Any],
) -> Mapping[str, Any]:
    manual_summary = _mapping(manual_resolution.get("summary"))
    duplicate_summary = _mapping(duplicate_resolution)
    duplicate_counts = _mapping(duplicate_summary.get("by_duplicate_resolution_status"))
    manager_text_review_rows = max(
        0,
        _int(summary.get("queue_manual_resolution_rows"))
        - _int(duplicate_counts.get("duplicate_contacts_merge_required"))
        - _int(duplicate_counts.get("contact_id_mismatch_requires_operator")),
    )
    waiting_counts = _mapping(waiting_work.get("counts"))
    return {
        "schema_version": "operator_leadership_snapshot_v1",
        "manager_task_total_rows": _int(summary.get("queue_manual_resolution_rows")),
        "manager_duplicate_merge_required_rows": _int(duplicate_counts.get("duplicate_contacts_merge_required")),
        "manager_contact_id_mismatch_rows": _int(duplicate_counts.get("contact_id_mismatch_requires_operator")),
        "manager_text_quality_review_rows": manager_text_review_rows,
        "manager_already_written_review_rows": _int(manual_summary.get("already_written_review_rows")),
        "duplicate_staff_task_rows": _int(duplicate_staff.get("task_rows")),
        "duplicate_candidate_contact_rows": _int(duplicate_summary.get("candidate_contact_rows")),
        "duplicate_recheck_status": duplicate_recheck.get("status") or "not_built",
        "duplicate_recheck_pending_rows": _int(duplicate_recheck.get("blocked_rows")),
        "duplicate_recheck_passed": bool(duplicate_recheck.get("passed")),
        "duplicate_after_staff_done_status": duplicate_after_staff_done.get("status") or "not_built",
        "duplicate_after_staff_done_candidate_rows": _int(duplicate_after_staff_done.get("candidate_rows")),
        "duplicate_after_staff_done_blocked_rows": _int(duplicate_after_staff_done.get("blocked_rows")),
        "waiting_work_status": waiting_work.get("status") or "not_built",
        "waiting_work_non_duplicate_candidate_rows": _int(waiting_counts.get("non_duplicate_live_candidate_rows")),
        "waiting_work_refresh_candidate_rows": _int(waiting_counts.get("refresh_candidate_rows")),
        "waiting_work_readback_missing_rows": _int(waiting_counts.get("readback_missing_rows")),
        "waiting_work_contact_id_mismatch_rows": _int(waiting_counts.get("contact_id_mismatch_rows")),
        "ready_rows": _int(summary.get("queue_ready_rows")),
        "dry_run_allowed_now": False,
        "resolved_dry_run_allowed_now": False,
        "live_write_allowed_now": False,
        "stage50_preflight_allowed": bool(stage_rollout.get("stage50_preflight_allowed")),
        "stage86_preflight_allowed": bool(stage_rollout.get("stage86_preflight_allowed")),
        "stage50_blocked": bool(stage_rollout.get("stage50_blocked")),
        "stage86_blocked": bool(stage_rollout.get("stage86_blocked")),
        "cleanup_candidate_rows": _int(cleanup_status.get("candidate_rows")),
        "cleanup_safe_to_quarantine_rows": _int(cleanup_status.get("safe_to_quarantine_rows")),
        "cleanup_requires_human_review_rows": _int(cleanup_status.get("requires_human_review_rows")),
    }


def _load_queue_rows(queue_summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    outputs = _mapping(_mapping(queue_summary.get("outputs")).get("buckets"))
    rows: list[Mapping[str, Any]] = []
    for bucket in QUEUE_LABELS:
        csv_path = _path_from_value(_mapping(outputs.get(bucket)).get("csv"))
        if not csv_path or not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
    return rows


def _find_latest_manual_resolution_summary(project_root: Path, queue_summary: Mapping[str, Any]) -> Optional[Path]:
    queue_root = _path_from_value(queue_summary.get("out_root"))
    candidates = sorted((project_root / "stable_runtime").glob("amo_manual_resolution_*/summary.json"))
    if not candidates:
        return None
    if queue_root is None:
        return candidates[-1]
    matches: list[Path] = []
    for path in candidates:
        payload = _load_json_if_exists(path)
        summary = _mapping(payload.get("summary"))
        if _path_from_value(summary.get("queue_root")) == queue_root:
            matches.append(path)
    return matches[-1] if matches else candidates[-1]


def _find_latest_duplicate_resolution_summary(project_root: Path) -> Optional[Path]:
    candidates = sorted((project_root / "stable_runtime").glob("amo_duplicate_resolution_*/summary.json"))
    return candidates[-1] if candidates else None


def _find_latest_summary(project_root: Path, pattern: str) -> Optional[Path]:
    candidates = sorted((project_root / "stable_runtime").glob(f"{pattern}/summary.json"))
    return candidates[-1] if candidates else None


def _operator_queue_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    bucket = str(row.get("queue_bucket") or "").strip()
    return {
        "queue_bucket": bucket,
        "status_ru": QUEUE_LABELS.get(bucket, bucket),
        "action_required_ru": QUEUE_ACTIONS.get(bucket, "Требуется ручная проверка статуса."),
        "source_row_index": row.get("source_row_index", ""),
        "phone": row.get("normalized_phone") or row.get("Телефон клиента") or "",
        "amo_contact_ids": row.get("source_amo_contact_ids") or row.get("AMO contact IDs") or "",
        "effective_contact_id": row.get("effective_contact_id") or "",
        "fio_parent": row.get("ФИО родителя") or "",
        "fio_child": row.get("ФИО ребенка") or "",
        "latest_call_date": row.get("Дата последнего свежего звонка") or "",
        "latest_call_type": row.get("Тип последнего свежего звонка") or "",
        "priority": row.get("Приоритет лида") or "",
        "sale_probability_percent": row.get("Вероятность продажи, %") or "",
        "next_step": row.get("Следующий шаг") or "",
        "queue_reason": row.get("queue_reason") or "",
        "dry_run_status": row.get("dry_run_status") or "",
        "written_status": row.get("written_status") or "",
        "manual_review_report": row.get("manual_review_report") or "",
    }


def _write_operator_queue_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fieldnames = [
        "queue_bucket",
        "status_ru",
        "action_required_ru",
        "source_row_index",
        "phone",
        "amo_contact_ids",
        "effective_contact_id",
        "fio_parent",
        "fio_child",
        "latest_call_date",
        "latest_call_type",
        "priority",
        "sale_probability_percent",
        "next_step",
        "queue_reason",
        "dry_run_status",
        "written_status",
        "manual_review_report",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_operator_status_markdown(status: Mapping[str, Any]) -> str:
    summary = _mapping(status.get("summary"))
    loop = _mapping(status.get("amo_production_loop"))
    queue = _mapping(status.get("crm_queue"))
    manual_resolution = _mapping(status.get("manual_resolution"))
    manual_summary = _mapping(manual_resolution.get("summary"))
    duplicate_resolution = _mapping(status.get("duplicate_resolution"))
    duplicate_summary = _mapping(duplicate_resolution.get("summary"))
    duplicate_staff = _mapping(status.get("duplicate_staff_tasks"))
    duplicate_staff_summary = _mapping(duplicate_staff.get("summary"))
    duplicate_recheck = _mapping(status.get("duplicate_post_merge_recheck"))
    duplicate_recheck_summary = _mapping(duplicate_recheck.get("summary"))
    duplicate_after_staff_done = _mapping(status.get("duplicate_after_staff_done"))
    duplicate_after_staff_done_summary = _mapping(duplicate_after_staff_done.get("summary"))
    waiting_work = _mapping(status.get("waiting_autonomous_work"))
    waiting_counts = _mapping(waiting_work.get("counts"))
    lines = [
        "# Mango Analyse operator status",
        "",
        f"Generated at: `{summary.get('generated_at')}`",
        "",
        "## Summary",
        "",
        f"- Runtime validation: `{summary.get('runtime_validation_ok')}`",
        f"- Call processing ready: `{summary.get('call_processing_ready')}`",
        f"- CRM live-write allowed now: `{summary.get('crm_writeback_live_allowed_now')}`",
        f"- Blocking status: `{summary.get('blocked')}` (`{summary.get('blocked_semantics')}`)",
        f"- Contact-id mismatch blocked rows: `{summary.get('contact_id_mismatch_blocked_rows')}`",
        f"- Canonical actionable calls: `{summary.get('canonical_actionable_calls')}`",
        f"- Missing ASR: `{summary.get('canonical_missing_asr')}`",
        f"- Missing Resolve+Analyze: `{summary.get('canonical_missing_ra')}`",
        f"- AMO-ready rows: `{summary.get('amo_ready_rows')}`",
        f"- Ready queue rows: `{summary.get('queue_ready_rows')}`",
        f"- Already written rows: `{summary.get('queue_already_written_rows')}`",
        f"- Manual-resolution rows: `{summary.get('queue_manual_resolution_rows')}`",
        f"- Duplicate merge required rows: `{summary.get('duplicate_merge_required_rows')}`",
        f"- Duplicate mismatch rows: `{summary.get('duplicate_contact_mismatch_rows')}`",
        f"- Duplicate staff task rows: `{summary.get('duplicate_staff_task_rows')}`",
        f"- Duplicate recheck passed: `{summary.get('duplicate_recheck_passed')}`",
        f"- Duplicate recheck blocked rows: `{summary.get('duplicate_recheck_blocked_rows')}`",
        f"- Duplicate after-staff-done status: `{summary.get('duplicate_after_staff_done_status')}`",
        f"- Duplicate after-staff-done candidate rows: `{summary.get('duplicate_after_staff_done_candidate_rows')}`",
        f"- Duplicate after-staff-done blocked rows: `{summary.get('duplicate_after_staff_done_blocked_rows')}`",
        f"- Waiting-work status: `{summary.get('waiting_work_status')}`",
        f"- Waiting-work dry-run prepared rows: `{summary.get('waiting_work_dry_run_prepared_rows')}`",
        f"- Waiting-work refresh candidates: `{summary.get('waiting_work_refresh_candidate_rows')}`",
        f"- Waiting-work readback missing rows: `{summary.get('waiting_work_readback_missing_rows')}`",
        f"- Waiting-work live-write allowed now: `{summary.get('waiting_work_live_write_allowed_now')}`",
        f"- Resolved live candidates: `{manual_summary.get('resolved_live_candidate_rows', 0)}`",
        "",
        "## AMO production loop",
        "",
        f"- Stage: `{loop.get('stage')}`",
        f"- Dry-run allowed now: `{loop.get('dry_run_allowed_now')}`",
        f"- Live write allowed now: `{loop.get('live_write_allowed_now')}`",
        f"- Blocking reasons: `{', '.join(loop.get('blocking_reasons') or []) or 'none'}`",
        "",
        "## CRM queue buckets",
        "",
    ]
    for bucket in _mapping(queue).get("buckets") or []:
        lines.append(f"- {bucket['label_ru']}: `{bucket['rows']}`. {bucket['action_ru']}")
    lines.extend(
        [
            "",
            "## AMO duplicate resolution",
            "",
            f"- Summary path: `{duplicate_resolution.get('summary_path') or 'not built'}`",
            f"- Review rows: `{duplicate_summary.get('review_rows', 0)}`",
            f"- Candidate contact rows: `{duplicate_summary.get('candidate_contact_rows', 0)}`",
            f"- Post-merge recheck rows: `{duplicate_summary.get('post_merge_recheck_rows', 0)}`",
            f"- Staff task pack: `{duplicate_staff.get('summary_path') or 'not built'}`",
            f"- Staff task rows: `{duplicate_staff_summary.get('task_rows', 0)}`",
            f"- Recheck gate: `{duplicate_recheck.get('summary_path') or 'not built'}`",
            f"- Recheck status: `{duplicate_recheck_summary.get('status', 'not built')}`",
            f"- Recheck passed: `{duplicate_recheck_summary.get('passed', False)}`",
            f"- Recheck blocked rows: `{duplicate_recheck_summary.get('blocked_rows', 0)}`",
            f"- After-staff-done pipeline: `{duplicate_after_staff_done.get('summary_path') or 'not built'}`",
        f"- After-staff-done status: `{duplicate_after_staff_done_summary.get('status', 'not built')}`",
        f"- After-staff-done candidate rows: `{duplicate_after_staff_done_summary.get('candidate_rows', 0)}`",
        f"- After-staff-done blocked rows: `{duplicate_after_staff_done_summary.get('blocked_rows', 0)}`",
        "",
        "## Waiting autonomous work",
        "",
        f"- Summary path: `{waiting_work.get('summary_path') or 'not built'}`",
        f"- Status: `{waiting_work.get('status') or 'not_built'}`",
        f"- Non-duplicate candidates: `{waiting_counts.get('non_duplicate_live_candidate_rows', 0)}`",
        f"- Refresh candidates: `{waiting_counts.get('refresh_candidate_rows', 0)}`",
        f"- Missing readback rows: `{waiting_counts.get('readback_missing_rows', 0)}`",
        f"- Contact-id mismatch rows: `{waiting_counts.get('contact_id_mismatch_rows', 0)}`",
        f"- Dry-run allowed when tunnel available: `{waiting_work.get('dry_run_allowed_when_tunnel_available', False)}`",
        f"- Live-write allowed now: `{waiting_work.get('live_write_allowed_now', False)}`",
        ]
    )
    lines.extend(["", "## Next actions", ""])
    for action in loop.get("next_operator_actions") or []:
        lines.append(f"- `{action['action']}`: {action['description_ru']} Rows: `{action['rows']}`.")
    lines.append("")
    return "\n".join(lines)


def render_operator_dashboard_html(status: Mapping[str, Any]) -> str:
    summary = _mapping(status.get("summary"))
    loop = _mapping(status.get("amo_production_loop"))
    queue = _mapping(status.get("crm_queue"))
    manual_resolution = _mapping(status.get("manual_resolution"))
    manual_summary = _mapping(manual_resolution.get("summary"))
    duplicate_resolution = _mapping(status.get("duplicate_resolution"))
    duplicate_summary = _mapping(duplicate_resolution.get("summary"))
    duplicate_staff = _mapping(status.get("duplicate_staff_tasks"))
    duplicate_staff_summary = _mapping(duplicate_staff.get("summary"))
    duplicate_recheck = _mapping(status.get("duplicate_post_merge_recheck"))
    duplicate_recheck_summary = _mapping(duplicate_recheck.get("summary"))
    duplicate_after_staff_done = _mapping(status.get("duplicate_after_staff_done"))
    duplicate_after_staff_done_summary = _mapping(duplicate_after_staff_done.get("summary"))
    waiting_work = _mapping(status.get("waiting_autonomous_work"))
    waiting_counts = _mapping(waiting_work.get("counts"))
    metric_cards = [
        ("Runtime", "OK" if summary.get("runtime_validation_ok") else "BLOCK", "green" if summary.get("runtime_validation_ok") else "red"),
        ("Blocking reasons", summary.get("production_blocking_reasons_count"), "orange" if _int(summary.get("production_blocking_reasons_count")) else "green"),
        ("Contact mismatch", summary.get("contact_id_mismatch_blocked_rows"), "orange" if _int(summary.get("contact_id_mismatch_blocked_rows")) else "green"),
        ("Calls", summary.get("canonical_actionable_calls"), "blue"),
        ("Missing ASR", summary.get("canonical_missing_asr"), "green" if _int(summary.get("canonical_missing_asr")) == 0 else "red"),
        ("Missing R+A", summary.get("canonical_missing_ra"), "green" if _int(summary.get("canonical_missing_ra")) == 0 else "red"),
        ("AMO ready", summary.get("amo_ready_rows"), "blue"),
        ("Ready live", summary.get("queue_ready_rows"), "green" if _int(summary.get("queue_ready_rows")) else "muted"),
        ("Written", summary.get("queue_already_written_rows"), "blue"),
        ("Manual queue", summary.get("queue_manual_resolution_rows"), "orange" if _int(summary.get("queue_manual_resolution_rows")) else "green"),
        ("Duplicate merge", summary.get("duplicate_merge_required_rows"), "orange" if _int(summary.get("duplicate_merge_required_rows")) else "green"),
        ("Mismatch", summary.get("duplicate_contact_mismatch_rows"), "orange" if _int(summary.get("duplicate_contact_mismatch_rows")) else "green"),
        ("Staff tasks", summary.get("duplicate_staff_task_rows"), "orange" if _int(summary.get("duplicate_staff_task_rows")) else "muted"),
        ("Recheck blocked", summary.get("duplicate_recheck_blocked_rows"), "orange" if _int(summary.get("duplicate_recheck_blocked_rows")) else "green"),
        ("After-staff candidates", summary.get("duplicate_after_staff_done_candidate_rows"), "green" if _int(summary.get("duplicate_after_staff_done_candidate_rows")) else "muted"),
        ("After-staff blocked", summary.get("duplicate_after_staff_done_blocked_rows"), "orange" if _int(summary.get("duplicate_after_staff_done_blocked_rows")) else "green"),
        ("Waiting dry-run", summary.get("waiting_work_dry_run_prepared_rows"), "green" if _int(summary.get("waiting_work_dry_run_prepared_rows")) else "muted"),
        ("Refresh", summary.get("waiting_work_refresh_candidate_rows"), "blue" if _int(summary.get("waiting_work_refresh_candidate_rows")) else "muted"),
        ("Readback missing", summary.get("waiting_work_readback_missing_rows"), "orange" if _int(summary.get("waiting_work_readback_missing_rows")) else "green"),
        ("Resolved candidates", manual_summary.get("resolved_live_candidate_rows", 0), "green" if _int(manual_summary.get("resolved_live_candidate_rows")) else "muted"),
    ]
    cards = "\n".join(
        f'<div class="card {color}"><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></div>'
        for label, value, color in metric_cards
    )
    bucket_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item['label_ru']))}</td>"
        f"<td>{html.escape(str(item['rows']))}</td>"
        f"<td>{html.escape(str(item['action_ru']))}</td>"
        "</tr>"
        for item in queue.get("buckets") or []
    )
    actions = "\n".join(
        f"<li><b>{html.escape(str(action['action']))}</b>: {html.escape(str(action['description_ru']))} <span>{html.escape(str(action['rows']))} rows</span></li>"
        for action in loop.get("next_operator_actions") or []
    )
    payload = html.escape(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mango Analyse Operator Status</title>
  <style>
    :root {{ --bg:#f6f3ec; --ink:#18212b; --muted:#667085; --line:#ddd4c4; --paper:#fffdf8; --blue:#2457a7; --green:#0d7a53; --orange:#a15c00; --red:#b42318; }}
    body {{ margin:0; background:linear-gradient(135deg,#f6f3ec,#eef4f8); color:var(--ink); font-family:ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ padding:28px 32px 18px; border-bottom:1px solid var(--line); background:rgba(255,253,248,.88); position:sticky; top:0; backdrop-filter:blur(10px); }}
    h1 {{ margin:0; font-size:28px; letter-spacing:-.03em; }}
    .sub {{ color:var(--muted); margin-top:6px; }}
    main {{ padding:24px 32px 48px; max-width:1360px; margin:auto; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
    .card {{ background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 14px 30px rgba(24,33,43,.06); }}
    .card span {{ display:block; color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; }}
    .card strong {{ display:block; font-size:28px; margin-top:8px; }}
    .green strong {{ color:var(--green); }} .orange strong {{ color:var(--orange); }} .red strong {{ color:var(--red); }} .blue strong {{ color:var(--blue); }} .muted strong {{ color:var(--muted); }}
    section {{ background:var(--paper); border:1px solid var(--line); border-radius:18px; margin-top:18px; overflow:hidden; box-shadow:0 14px 30px rgba(24,33,43,.06); }}
    section h2 {{ margin:0; padding:18px 20px; border-bottom:1px solid var(--line); font-size:18px; }}
    table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; padding:12px 14px; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
    li {{ margin:10px 0; }} li span {{ color:var(--muted); }} pre {{ margin:0; padding:18px; max-height:420px; overflow:auto; background:#111827; color:#e5e7eb; }}
    @media(max-width:720px){{ header,main{{padding-left:16px;padding-right:16px}} }}
  </style>
</head>
<body>
<header>
  <h1>Mango Analyse Operator Status</h1>
  <div class="sub">Read-only runtime health, CRM queue and AMO production-loop status. Generated: {html.escape(str(summary.get('generated_at')))}</div>
</header>
<main>
  <div class="grid">{cards}</div>
  <section><h2>AMO production loop</h2><table><tbody>
    <tr><th>Stage</th><td>{html.escape(str(loop.get('stage')))}</td></tr>
    <tr><th>Dry-run allowed</th><td>{html.escape(str(loop.get('dry_run_allowed_now')))}</td></tr>
    <tr><th>Live write allowed</th><td>{html.escape(str(loop.get('live_write_allowed_now')))}</td></tr>
    <tr><th>Blockers</th><td>{html.escape(', '.join(loop.get('blocking_reasons') or []) or 'none')}</td></tr>
  </tbody></table></section>
  <section><h2>CRM queue buckets</h2><table><thead><tr><th>Bucket</th><th>Rows</th><th>Action</th></tr></thead><tbody>{bucket_rows}</tbody></table></section>
  <section><h2>Manual resolution</h2><table><tbody>
    <tr><th>Summary path</th><td>{html.escape(str(manual_resolution.get('summary_path') or 'not built'))}</td></tr>
    <tr><th>Review rows</th><td>{html.escape(str(manual_summary.get('review_rows', 0)))}</td></tr>
    <tr><th>Accepted rows</th><td>{html.escape(str(manual_summary.get('accepted_rows', 0)))}</td></tr>
    <tr><th>Resolved live candidates</th><td>{html.escape(str(manual_summary.get('resolved_live_candidate_rows', 0)))}</td></tr>
    <tr><th>Needs human</th><td>{html.escape(str(manual_summary.get('needs_human_rows', 0)))}</td></tr>
    <tr><th>Still blocked</th><td>{html.escape(str(manual_summary.get('still_blocked_rows', 0)))}</td></tr>
  </tbody></table></section>
  <section><h2>AMO duplicate resolution</h2><table><tbody>
    <tr><th>Summary path</th><td>{html.escape(str(duplicate_resolution.get('summary_path') or 'not built'))}</td></tr>
    <tr><th>Review rows</th><td>{html.escape(str(duplicate_summary.get('review_rows', 0)))}</td></tr>
    <tr><th>Candidate contact rows</th><td>{html.escape(str(duplicate_summary.get('candidate_contact_rows', 0)))}</td></tr>
    <tr><th>Post-merge recheck rows</th><td>{html.escape(str(duplicate_summary.get('post_merge_recheck_rows', 0)))}</td></tr>
    <tr><th>Staff task pack</th><td>{html.escape(str(duplicate_staff.get('summary_path') or 'not built'))}</td></tr>
    <tr><th>Staff task rows</th><td>{html.escape(str(duplicate_staff_summary.get('task_rows', 0)))}</td></tr>
    <tr><th>Recheck gate</th><td>{html.escape(str(duplicate_recheck.get('summary_path') or 'not built'))}</td></tr>
    <tr><th>Recheck status</th><td>{html.escape(str(duplicate_recheck_summary.get('status', 'not built')))}</td></tr>
    <tr><th>Recheck passed</th><td>{html.escape(str(duplicate_recheck_summary.get('passed', False)))}</td></tr>
    <tr><th>Recheck blocked rows</th><td>{html.escape(str(duplicate_recheck_summary.get('blocked_rows', 0)))}</td></tr>
    <tr><th>After-staff pipeline</th><td>{html.escape(str(duplicate_after_staff_done.get('summary_path') or 'not built'))}</td></tr>
    <tr><th>After-staff status</th><td>{html.escape(str(duplicate_after_staff_done_summary.get('status', 'not built')))}</td></tr>
    <tr><th>After-staff candidates</th><td>{html.escape(str(duplicate_after_staff_done_summary.get('candidate_rows', 0)))}</td></tr>
    <tr><th>After-staff blocked</th><td>{html.escape(str(duplicate_after_staff_done_summary.get('blocked_rows', 0)))}</td></tr>
  </tbody></table></section>
  <section><h2>Waiting autonomous work</h2><table><tbody>
    <tr><th>Summary path</th><td>{html.escape(str(waiting_work.get('summary_path') or 'not built'))}</td></tr>
    <tr><th>Status</th><td>{html.escape(str(waiting_work.get('status') or 'not_built'))}</td></tr>
    <tr><th>Non-duplicate candidates</th><td>{html.escape(str(waiting_counts.get('non_duplicate_live_candidate_rows', 0)))}</td></tr>
    <tr><th>Refresh candidates</th><td>{html.escape(str(waiting_counts.get('refresh_candidate_rows', 0)))}</td></tr>
    <tr><th>Missing readback rows</th><td>{html.escape(str(waiting_counts.get('readback_missing_rows', 0)))}</td></tr>
    <tr><th>Contact-id mismatch rows</th><td>{html.escape(str(waiting_counts.get('contact_id_mismatch_rows', 0)))}</td></tr>
    <tr><th>Dry-run allowed when tunnel available</th><td>{html.escape(str(waiting_work.get('dry_run_allowed_when_tunnel_available', False)))}</td></tr>
    <tr><th>Live-write allowed now</th><td>{html.escape(str(waiting_work.get('live_write_allowed_now', False)))}</td></tr>
  </tbody></table></section>
  <section><h2>Next operator actions</h2><ul>{actions}</ul></section>
  <section><h2>Raw status JSON</h2><pre>{payload}</pre></section>
</main>
</body>
</html>
"""


def _load_json_if_exists(path: Optional[Path]) -> Mapping[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_optional(project_root: Path, path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def _path_from_value(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve(strict=False)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "OPERATOR_STATUS_SCHEMA_VERSION",
    "DEFAULT_OPERATOR_STATUS_ROOT",
    "build_operator_status",
    "render_operator_dashboard_html",
    "render_operator_status_markdown",
]
