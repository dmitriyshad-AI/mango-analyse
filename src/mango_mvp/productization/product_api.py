from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from mango_mvp.productization.crm_tallanto_mapping_preview import build_crm_tallanto_mapping_preview
from mango_mvp.productization.crm_writeback_preview import build_crm_writeback_preview
from mango_mvp.productization.saas_stage_gates import build_saas_stage_gates_report
from mango_mvp.productization.saas_demo_contracts import build_dashboard_demo_readiness
from mango_mvp.productization.scheduler_control_plane import build_scheduler_control_plane_report
from mango_mvp.productization.scheduler_health import build_scheduler_health_report
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


PRODUCT_API_SCHEMA_VERSION = "product_api_readonly_v1"
APPLIANCE_DASHBOARD_SCHEMA_VERSION = "appliance_dashboard_v1"


@dataclass(frozen=True)
class ProductApiSummary:
    schema_version: str
    product_root: str
    product_db_path: str
    endpoints: int
    read_only: bool
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


class ProductApiFacade:
    """Read-only facade for the product appliance.

    This is a dependency-free API skeleton. A future FastAPI layer should call
    these methods instead of reading SQLite files or runtime artifacts directly.
    """

    def __init__(self, product_root: Path, product_db_path: Path, workspace_root: Optional[Path] = None) -> None:
        self.product_root = product_root.resolve(strict=False)
        self.product_db_path = product_db_path.resolve(strict=False)
        self.workspace_root = (workspace_root or Path.cwd()).resolve(strict=False)
        guard_product_api_paths(self.product_root, self.product_db_path)

    def dashboard_summary(self) -> Mapping[str, Any]:
        snapshot = self.product_db_snapshot()
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /dashboard/summary",
            "summary": snapshot["summary"],
            "health": {
                "product_db_present": self.product_db_path.exists(),
                "validation_ok": snapshot["summary"]["validation_ok"],
                "blocked": snapshot["summary"]["blocked"],
                "warnings": snapshot["summary"]["warnings"],
            },
            "actions": read_only_actions(),
        }

    def appliance_dashboard(
        self,
        *,
        capture_limit: int = 25,
        scheduler_limit: int = 25,
        capture_status: str = "",
        manager_ref: str = "",
        q: str = "",
        scheduler_status: str = "",
        scheduler_job_type: str = "",
    ) -> Mapping[str, Any]:
        if capture_limit < 1:
            raise ValueError("capture_limit must be positive")
        if scheduler_limit < 1:
            raise ValueError("scheduler_limit must be positive")

        dashboard = self.dashboard_summary()
        capture = self.capture_recent(
            limit=capture_limit,
            status=capture_status,
            manager_ref=manager_ref,
            q=q,
        )
        queue = self.processing_queue(limit=capture_limit)
        scheduler = self.scheduler_runs(
            limit=scheduler_limit,
            status=scheduler_status,
            job_type=scheduler_job_type,
        )
        scheduler_health = self.scheduler_health()
        scheduler_control = self.scheduler_control_plane()
        lifecycle = self.lifecycle_readiness()
        asr_gate = self.asr_gate_status()
        writeback = self.writeback_previews(limit=capture_limit)
        operator_status = self.operator_runtime_status()
        waiting_work = self.waiting_autonomous_work_status()
        crm_mapping = self.crm_mapping_preview(limit=capture_limit)
        knowledge = self.knowledge_playbook()
        settings = self.settings_adapters()
        capture_items = capture.get("items") if isinstance(capture.get("items"), list) else []
        scheduler_items = scheduler.get("items") if isinstance(scheduler.get("items"), list) else []
        panels = {
            "dashboard": dashboard,
            "capture": {
                "endpoint": capture.get("endpoint"),
                "limit": capture.get("limit"),
                "items": capture_items,
                "total_visible": len(capture_items),
            },
            "processing_queue": queue,
            "scheduler": {
                "endpoint": scheduler.get("endpoint"),
                "limit": scheduler.get("limit"),
                "items": scheduler_items,
                "total_visible": len(scheduler_items),
                "health": scheduler_health,
                "control_plane": scheduler_control,
            },
            "lifecycle": lifecycle,
            "writeback": writeback,
            "operator_status": operator_status,
            "manual_resolution": self.manual_resolution_status(),
            "waiting_autonomous_work": waiting_work,
            "dry_run_readiness": self.writeback_dry_run_readiness(),
            "leadership_snapshot": operator_status.get("leadership_snapshot") or {},
            "stage_rollout": operator_status.get("stage_rollout") or {},
            "cleanup": operator_status.get("cleanup") or {},
            "crm_mapping": crm_mapping,
            "gates": {
                "asr": asr_gate,
                "run_asr": False,
                "run_ra": False,
                "write_runtime_db": False,
            },
            "knowledge": knowledge,
            "settings": settings,
        }
        demo_readiness = build_dashboard_demo_readiness(
            product_root=self.product_root,
            product_db_path=self.product_db_path,
            panels=panels,
        )
        panels["demo_readiness"] = demo_readiness

        db_summary = mapping_or_empty(dashboard.get("summary"))
        operator_summary = mapping_or_empty(operator_status.get("summary"))
        operator_loop = mapping_or_empty(operator_status.get("amo_production_loop"))
        leadership_snapshot = mapping_or_empty(operator_status.get("leadership_snapshot"))
        stage_rollout = mapping_or_empty(operator_status.get("stage_rollout"))
        cleanup_status = mapping_or_empty(operator_status.get("cleanup"))
        ready_capture = int(db_summary.get("capture_inbox_ready") or 0)
        blocked = int(db_summary.get("blocked") or 0)
        warnings = int(db_summary.get("warnings") or 0)
        writeback_blocked = list(writeback.get("blocked_reasons") or [])
        status = "ready" if bool(db_summary.get("validation_ok")) and blocked == 0 else "blocked"

        return {
            "schema_version": APPLIANCE_DASHBOARD_SCHEMA_VERSION,
            "endpoint": "GET /dashboard/appliance",
            "status": status,
            "summary": {
                "product_db_present": bool(dashboard.get("health", {}).get("product_db_present")),
                "validation_ok": bool(db_summary.get("validation_ok")),
                "blocked": blocked,
                "warnings": warnings,
                "tenants": int(db_summary.get("tenants") or 0),
                "product_calls": int(db_summary.get("product_calls") or 0),
                "capture_inbox_items": int(db_summary.get("capture_inbox_items") or 0),
                "capture_inbox_ready": ready_capture,
                "capture_inbox_blocked": int(db_summary.get("capture_inbox_blocked") or 0),
                "job_runs": int(db_summary.get("job_runs") or 0),
                "scheduler_due_jobs": int(mapping_or_empty(scheduler_health.get("summary")).get("due_jobs") or 0),
                "scheduler_failed_jobs": int(mapping_or_empty(scheduler_health.get("summary")).get("failed_jobs") or 0),
                "pending_owner_mappings": int(db_summary.get("pending_owner_mappings") or 0),
                "writeback_mode": writeback.get("current_mode"),
                "writeback_blocked_reasons": writeback_blocked,
                "operator_runtime_validation_ok": bool(operator_summary.get("runtime_validation_ok")),
                "operator_queue_ready_rows": int(operator_summary.get("queue_ready_rows") or 0),
                "operator_queue_manual_resolution_rows": int(operator_summary.get("queue_manual_resolution_rows") or 0),
                "operator_duplicate_merge_required_rows": int(operator_summary.get("duplicate_merge_required_rows") or 0),
                "operator_duplicate_contact_mismatch_rows": int(operator_summary.get("duplicate_contact_mismatch_rows") or 0),
                "operator_duplicate_staff_task_rows": int(operator_summary.get("duplicate_staff_task_rows") or 0),
                "operator_duplicate_recheck_blocked_rows": int(operator_summary.get("duplicate_recheck_blocked_rows") or 0),
                "operator_duplicate_recheck_passed": bool(operator_summary.get("duplicate_recheck_passed")),
                "operator_waiting_work_status": operator_summary.get("waiting_work_status") or "not_built",
                "operator_waiting_work_dry_run_prepared_rows": int(operator_summary.get("waiting_work_dry_run_prepared_rows") or 0),
                "operator_waiting_work_refresh_candidate_rows": int(operator_summary.get("waiting_work_refresh_candidate_rows") or 0),
                "operator_waiting_work_readback_missing_rows": int(operator_summary.get("waiting_work_readback_missing_rows") or 0),
                "operator_waiting_work_live_write_allowed_now": bool(operator_summary.get("waiting_work_live_write_allowed_now")),
                "manager_task_total_rows": int(leadership_snapshot.get("manager_task_total_rows") or 0),
                "stage50_preflight_allowed": bool(stage_rollout.get("stage50_preflight_allowed")),
                "stage86_preflight_allowed": bool(stage_rollout.get("stage86_preflight_allowed")),
                "cleanup_candidate_rows": int(cleanup_status.get("candidate_rows") or 0),
                "cleanup_safe_to_quarantine_rows": int(cleanup_status.get("safe_to_quarantine_rows") or 0),
                "cleanup_requires_human_review_rows": int(cleanup_status.get("requires_human_review_rows") or 0),
                "operator_any_dry_run_allowed_now": bool(operator_loop.get("any_dry_run_allowed_now")),
                "operator_waiting_work_dry_run_allowed_now": bool(operator_loop.get("waiting_work_dry_run_allowed_now")),
                "operator_dry_run_allowed_now": bool(operator_loop.get("any_dry_run_allowed_now") or operator_loop.get("dry_run_allowed_now") or operator_loop.get("resolved_dry_run_allowed_now")),
                "operator_live_write_allowed_now": False,
                "operator_live_write_approval_required": bool(operator_loop.get("live_write_approval_required", True)),
                "demo_snapshot_files": int(mapping_or_empty(demo_readiness.get("summary")).get("snapshot_files") or 0),
                "demo_artifacts": int(mapping_or_empty(demo_readiness.get("summary")).get("demo_artifacts") or 0),
            },
            "filters": {
                "capture_status": clean(capture_status) or None,
                "manager_ref": clean(manager_ref) or None,
                "q": clean(q) or None,
                "scheduler_status": clean(scheduler_status) or None,
                "scheduler_job_type": clean(scheduler_job_type) or None,
            },
            "panels": panels,
            "navigation": [
                {"id": "summary", "label": "Summary"},
                {"id": "capture", "label": "Capture"},
                {"id": "scheduler", "label": "Scheduler"},
                {"id": "lifecycle", "label": "Lifecycle"},
                {"id": "writeback", "label": "Writeback"},
                {"id": "waiting_work", "label": "Waiting work"},
                {"id": "demo_readiness", "label": "Demo"},
                {"id": "knowledge", "label": "Knowledge"},
                {"id": "settings", "label": "Settings"},
            ],
            "actions": read_only_actions(),
            "safety": {
                "ui_mode": "read_only",
                "write_crm": False,
                "write_tallanto": False,
                "run_asr": False,
                "run_ra": False,
                "write_runtime_db": False,
                "blocked_actions": ["download_audio", "run_asr", "run_ra", "write_crm", "write_runtime_db"],
            },
        }

    def capture_recent(
        self,
        limit: int = 50,
        *,
        status: str = "",
        manager_ref: str = "",
        q: str = "",
    ) -> Mapping[str, Any]:
        if limit < 1:
            raise ValueError("limit must be positive")
        clauses = []
        params: list[Any] = []
        status = bounded_filter_text(status, "status")
        manager_ref = bounded_filter_text(manager_ref, "manager_ref")
        q = bounded_filter_text(q, "q")
        if status:
            clauses.append("status = ?")
            params.append(status)
        if manager_ref:
            clauses.append("manager_ref = ?")
            params.append(manager_ref)
        if q:
            like = like_value(q)
            clauses.append(
                """
                (
                  event_key LIKE ? ESCAPE '\\'
                  OR provider_call_id LIKE ? ESCAPE '\\'
                  OR recording_ref LIKE ? ESCAPE '\\'
                  OR client_phone LIKE ? ESCAPE '\\'
                  OR manager_ref LIKE ? ESCAPE '\\'
                )
                """
            )
            params.extend([like, like, like, like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._fetch_rows(
            f"""
            SELECT id, tenant_id, provider, event_key, provider_call_id, status,
                   started_at, direction, client_phone, manager_ref, recording_ref, audio_ref,
                   raw_payload_ref, enqueue_count, last_seen_at
              FROM capture_inbox_items
             {where}
             ORDER BY last_seen_at DESC, id DESC
             LIMIT ?
            """,
            (*params, int(limit)),
            required_table="capture_inbox_items",
        )
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /capture/recent",
            "limit": limit,
            "filters": {"status": status or None, "manager_ref": manager_ref or None, "q": q or None},
            "items": rows,
            "actions": read_only_actions(),
        }

    def scheduler_runs(self, limit: int = 50, *, status: str = "", job_type: str = "") -> Mapping[str, Any]:
        if limit < 1:
            raise ValueError("limit must be positive")
        clauses = []
        params: list[Any] = []
        status = bounded_filter_text(status, "status")
        job_type = bounded_filter_text(job_type, "job_type")
        if status:
            clauses.append("status = ?")
            params.append(status)
        if job_type:
            clauses.append("job_type = ?")
            params.append(job_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._fetch_rows(
            f"""
            SELECT id, job_type, tenant_id, status, planned_at,
                   scheduled_for, next_run_at, attempt_count, max_attempts,
                   lock_owner, lock_expires_at, heartbeat_at, output_ref
              FROM job_runs
             {where}
             ORDER BY id DESC
             LIMIT ?
            """,
            (*params, int(limit)),
            required_table="job_runs",
        )
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /scheduler/runs",
            "limit": limit,
            "filters": {"status": status or None, "job_type": job_type or None},
            "items": rows,
            "actions": read_only_actions(),
        }

    def asr_gate_status(self) -> Mapping[str, Any]:
        stage25_dir = self.product_root / "asr_worker_sandbox_execution_request_stage25"
        audit_path = stage25_dir / "asr_worker_sandbox_execution_request_stage25_audit.json"
        request_path = stage25_dir / "asr_worker_sandbox_execution_request_stage25.json"
        payload = load_json_if_exists(audit_path) or load_json_if_exists(request_path) or {}
        summary = mapping_or_empty(payload.get("summary"))
        execution_request = mapping_or_empty(payload.get("execution_request"))
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /asr/gates",
            "stage25": {
                "audit_path": str(audit_path) if audit_path.exists() else None,
                "request_path": str(request_path) if request_path.exists() else None,
                "approval_packet_ref": summary.get("approval_packet_ref") or execution_request.get("approval_packet_ref"),
                "execution_request_ready": bool(summary.get("execution_request_ready") or execution_request.get("execution_request_ready")),
                "dispatch_allowed": False,
                "run_asr": False,
                "write_transcripts": False,
                "missing_or_invalid_reasons": mapping_or_empty(payload.get("approval")).get("missing_or_invalid_reasons") or [],
            },
            "actions": read_only_actions(),
        }

    def writeback_previews(self, limit: int = 25, stage: str = "batch_10") -> Mapping[str, Any]:
        if limit < 1:
            raise ValueError("limit must be positive")
        stage = clean(stage) or "batch_10"
        if self.product_db_path.exists():
            try:
                preview = build_crm_writeback_preview(
                    product_db_path=self.product_db_path,
                    product_root=self.product_root,
                    stage=stage,
                    limit=limit,
                    crm_snapshot_path=default_crm_snapshot_path(self.product_root),
                )
                return {
                    "schema_version": PRODUCT_API_SCHEMA_VERSION,
                    "endpoint": "GET /writeback/previews",
                    "current_mode": "preview_only",
                    "write_crm": False,
                    "write_tallanto": False,
                    "preview": preview,
                    "blocked_reasons": writeback_blocked_reasons(preview),
                    "required_sequence": ["dry_run_diff", "human_review", "pilot_50", "pilot_300", "full_rollout"],
                    "actions": read_only_actions(),
                }
            except Exception as exc:
                return {
                    "schema_version": PRODUCT_API_SCHEMA_VERSION,
                    "endpoint": "GET /writeback/previews",
                    "current_mode": "preview_only",
                    "write_crm": False,
                    "write_tallanto": False,
                    "preview": None,
                    "blocked_reasons": ["writeback_preview_unavailable"],
                    "error": str(exc),
                    "required_sequence": ["dry_run_diff", "human_review", "pilot_50", "pilot_300", "full_rollout"],
                    "actions": read_only_actions(),
                }
        snapshot = self.product_db_snapshot()
        pending = int(snapshot["summary"].get("pending_owner_mappings") or 0)
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /writeback/previews",
            "current_mode": "preview_only",
            "write_crm": False,
            "write_tallanto": False,
            "blocked_reasons": ["pending_owner_mappings"] if pending else [],
            "required_sequence": ["dry_run_diff", "human_review", "pilot_50", "pilot_300", "full_rollout"],
            "actions": read_only_actions(),
        }

    def operator_runtime_status(self) -> Mapping[str, Any]:
        status_path = latest_operator_status_path(self.workspace_root)
        payload = load_json_if_exists(status_path)
        if not isinstance(payload, Mapping) or not payload:
            return {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "endpoint": "GET /operator/status",
                "status": "missing_operator_status_artifact",
                "summary": {"validation_ok": False, "blocked": 1, "warnings": 0},
                "path": str(status_path),
                "actions": read_only_actions(),
            }
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /operator/status",
            "status": "loaded",
            "path": str(status_path),
            **payload,
            "actions": read_only_actions(),
        }

    def manual_resolution_status(self) -> Mapping[str, Any]:
        operator = self.operator_runtime_status()
        manual = mapping_or_empty(operator.get("manual_resolution"))
        duplicate = mapping_or_empty(operator.get("duplicate_resolution"))
        duplicate_staff = mapping_or_empty(operator.get("duplicate_staff_tasks"))
        duplicate_recheck = mapping_or_empty(operator.get("duplicate_post_merge_recheck"))
        duplicate_after_staff_done = mapping_or_empty(operator.get("duplicate_after_staff_done"))
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /manual-resolution/status",
            "status": "loaded" if manual else "missing_manual_resolution_status",
            "manual_resolution": manual,
            "duplicate_resolution": duplicate,
            "duplicate_staff_tasks": duplicate_staff,
            "duplicate_post_merge_recheck": duplicate_recheck,
            "duplicate_after_staff_done": duplicate_after_staff_done,
            "summary": {
                "manual_review_rows": int(mapping_or_empty(manual.get("summary")).get("review_rows") or 0),
                "manual_resolved_live_candidates": int(mapping_or_empty(manual.get("summary")).get("resolved_live_candidate_rows") or 0),
                "duplicate_merge_required_rows": int(mapping_or_empty(duplicate.get("summary")).get("by_duplicate_resolution_status", {}).get("duplicate_contacts_merge_required") or 0),
                "duplicate_contact_mismatch_rows": int(mapping_or_empty(duplicate.get("summary")).get("by_duplicate_resolution_status", {}).get("contact_id_mismatch_requires_operator") or 0),
                "duplicate_staff_task_rows": int(mapping_or_empty(duplicate_staff.get("summary")).get("task_rows") or 0),
                "duplicate_recheck_blocked_rows": int(mapping_or_empty(duplicate_recheck.get("summary")).get("blocked_rows") or 0),
                "duplicate_recheck_passed": bool(mapping_or_empty(duplicate_recheck.get("summary")).get("passed")),
                "duplicate_after_staff_done_status": mapping_or_empty(duplicate_after_staff_done.get("summary")).get("status") or "not_built",
                "duplicate_after_staff_done_candidate_rows": int(mapping_or_empty(duplicate_after_staff_done.get("summary")).get("candidate_rows") or 0),
                "duplicate_after_staff_done_blocked_rows": int(mapping_or_empty(duplicate_after_staff_done.get("summary")).get("blocked_rows") or 0),
            },
            "write_crm": False,
            "actions": read_only_actions(),
        }

    def waiting_autonomous_work_status(self) -> Mapping[str, Any]:
        operator = self.operator_runtime_status()
        waiting = mapping_or_empty(operator.get("waiting_autonomous_work"))
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /waiting-work/status",
            "status": "loaded" if waiting else "missing_waiting_autonomous_work_status",
            "waiting_autonomous_work": waiting,
            "summary": {
                "status": waiting.get("status") or "not_built",
                "dry_run_allowed_when_tunnel_available": bool(waiting.get("dry_run_allowed_when_tunnel_available")),
                "live_write_allowed_now": False,
                **mapping_or_empty(waiting.get("counts")),
            },
            "required_sequence": [
                "quality_gate",
                "real_tunnel_dry_run",
                "claude_or_operator_audit",
                "explicit_live_approval",
                "staged_live_write",
                "post_writeback_readback",
            ],
            "write_crm": False,
            "actions": read_only_actions(),
        }

    def writeback_dry_run_readiness(self) -> Mapping[str, Any]:
        operator = self.operator_runtime_status()
        loop = mapping_or_empty(operator.get("amo_production_loop"))
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /writeback/dry-run-readiness",
            "status": loop.get("stage") or "missing_operator_production_loop",
            "dry_run_allowed_now": bool(loop.get("any_dry_run_allowed_now") or loop.get("dry_run_allowed_now") or loop.get("resolved_dry_run_allowed_now")),
            "waiting_work_dry_run_allowed_now": bool(loop.get("waiting_work_dry_run_allowed_now")),
            "live_write_allowed_now": False,
            "live_write_approval_required": True,
            "production_loop": loop,
            "actions": read_only_actions(),
        }

    def scheduler_health(self) -> Mapping[str, Any]:
        if not self.product_db_path.exists():
            return {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "endpoint": "GET /scheduler/health",
                "summary": {"validation_ok": False, "blocked": 1, "warnings": 0, "reason": "product_db_missing"},
                "actions": read_only_actions(),
            }
        try:
            report = build_scheduler_health_report(
                product_db_path=self.product_db_path,
                product_root=self.product_root,
            )
            return {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "endpoint": "GET /scheduler/health",
                **report,
                "actions": read_only_actions(),
            }
        except Exception as exc:
            return {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "endpoint": "GET /scheduler/health",
                "summary": {"validation_ok": False, "blocked": 1, "warnings": 0, "error": str(exc)},
                "actions": read_only_actions(),
            }

    def scheduler_control_plane(self) -> Mapping[str, Any]:
        if not self.product_db_path.exists():
            return {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "endpoint": "GET /scheduler/control-plane",
                "summary": {"validation_ok": False, "blocked": 1, "warnings": 0, "reason": "product_db_missing"},
                "actions": read_only_actions(),
            }
        try:
            report = build_scheduler_control_plane_report(
                product_db_path=self.product_db_path,
                product_root=self.product_root,
            )
            return {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "endpoint": "GET /scheduler/control-plane",
                **report,
                "actions": read_only_actions(),
            }
        except Exception as exc:
            return {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "endpoint": "GET /scheduler/control-plane",
                "summary": {"validation_ok": False, "blocked": 1, "warnings": 0, "error": str(exc)},
                "actions": read_only_actions(),
            }

    def crm_mapping_preview(self, limit: int = 50) -> Mapping[str, Any]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not self.product_db_path.exists():
            return {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "endpoint": "GET /crm/mapping-preview",
                "summary": {"validation_ok": False, "blocked": 1, "warnings": 0, "reason": "product_db_missing"},
                "actions": read_only_actions(),
            }
        try:
            report = build_crm_tallanto_mapping_preview(
                product_db_path=self.product_db_path,
                product_root=self.product_root,
                limit=limit,
            )
            return {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "endpoint": "GET /crm/mapping-preview",
                **report,
                "actions": read_only_actions(),
            }
        except Exception as exc:
            return {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "endpoint": "GET /crm/mapping-preview",
                "summary": {"validation_ok": False, "blocked": 1, "warnings": 0, "error": str(exc)},
                "actions": read_only_actions(),
            }

    def lifecycle_readiness(self) -> Mapping[str, Any]:
        candidates = sorted(self.product_root.glob("processing_lifecycle_stage5/*.json"))
        payload = load_json_if_exists(candidates[-1]) if candidates else None
        if isinstance(payload, Mapping):
            return {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "endpoint": "GET /processing/lifecycle",
                "mode": "report_file",
                "report_path": str(candidates[-1]),
                "report": payload,
                "actions": read_only_actions(),
            }
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /processing/lifecycle",
            "mode": "waiting_for_lifecycle_report",
            "blocked_reasons": ["processing_lifecycle_report_missing"],
            "run_asr": False,
            "run_ra": False,
            "write_runtime_db": False,
            "actions": read_only_actions(),
        }

    def processing_queue(self, limit: int = 50) -> Mapping[str, Any]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self._fetch_rows(
            """
            SELECT id, tenant_id, provider, event_key, provider_call_id, status,
                   started_at, manager_ref, recording_ref, audio_ref,
                   raw_payload_ref, enqueue_count, last_seen_at
              FROM capture_inbox_items
             WHERE status = 'ready_for_capture'
             ORDER BY started_at DESC, id DESC
             LIMIT ?
            """,
            (int(limit),),
            required_table="capture_inbox_items",
        )
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /queues/processing",
            "limit": limit,
            "items": rows,
            "queue_policy": {
                "auto_trigger_enabled": False,
                "requires_asr_gate": True,
                "run_asr": False,
                "write_runtime_db": False,
            },
            "actions": read_only_actions(),
        }

    def knowledge_playbook(self) -> Mapping[str, Any]:
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /knowledge/playbook",
            "current_mode": "schema_only",
            "entities": [
                "client_identity",
                "opportunity",
                "chain_event",
                "sales_moment",
                "response_quality_score",
                "outcome_link",
                "playbook_item",
            ],
            "items": [],
            "blocked_reasons": ["client_chain_layer_not_materialized", "outcome_linker_not_materialized"],
            "actions": read_only_actions(),
        }

    def settings_adapters(self) -> Mapping[str, Any]:
        return {
            "schema_version": PRODUCT_API_SCHEMA_VERSION,
            "endpoint": "GET /settings/adapters",
            "adapters": {
                "telephony": {
                    "primary": "mango",
                    "credentials_ref": ["env:MANGO_OFFICE_API_KEY", "env:MANGO_OFFICE_API_SALT"],
                    "write_capable": False,
                },
                "crm": {
                    "primary": "amocrm",
                    "write_mode": "preview_only",
                    "write_crm": False,
                },
                "database": {
                    "profile": "sqlite_single_writer_v1",
                    "product_db_path": str(self.product_db_path),
                },
            },
            "actions": read_only_actions(),
        }

    def saas_stage_gates(self) -> Mapping[str, Any]:
        return build_saas_stage_gates_report(
            product_root=self.product_root,
            product_db_path=self.product_db_path,
            workspace_root=self.workspace_root,
        )

    def product_db_snapshot(self) -> Mapping[str, Any]:
        if not self.product_db_path.exists():
            return {"summary": empty_db_summary(self.product_db_path)}
        with self.connect_ro() as con:
            tables = {name: count_relation(con, name) for name in PRODUCT_DB_TABLES}
            missing_relations = [name for name in PRODUCT_DB_TABLES if not relation_exists(con, name)]
            missing_migrations = missing_required_migrations(con)
            blocked = len(missing_relations) + len(missing_migrations)
            summary = {
                "schema_version": PRODUCT_API_SCHEMA_VERSION,
                "db_path": str(self.product_db_path),
                "schema_migrations": tables["schema_migrations"],
                "tenants": tables["tenants"],
                "product_calls": tables["product_calls"],
                "job_runs": tables["job_runs"],
                "capture_inbox_items": tables["capture_inbox_items"],
                "capture_inbox_ready": scalar_int_if_relation(
                    con,
                    "capture_inbox_items",
                    "select count(*) from capture_inbox_items where status = 'ready_for_capture'",
                ),
                "capture_inbox_blocked": scalar_int_if_relation(
                    con,
                    "capture_inbox_items",
                    "select count(*) from capture_inbox_items where status like 'blocked%'",
                ),
                "pending_owner_mappings": scalar_int_if_relation(
                    con,
                    "tenant_manager_owner_map",
                    "select count(*) from tenant_manager_owner_map where decision_status != 'confirmed_candidate'",
                ),
                "retention_policies": tables["retention_policies"],
                "validation_ok": blocked == 0,
                "blocked": blocked,
                "blocked_reasons": {
                    "missing_relations": missing_relations,
                    "missing_required_migrations": missing_migrations,
                },
                "warnings": scalar_int_if_relation(
                    con,
                    "tenant_manager_owner_map",
                    "select count(*) from tenant_manager_owner_map where decision_status != 'confirmed_candidate'",
                ),
            }
        return {"summary": summary, "tables": tables}

    def connect_ro(self) -> sqlite3.Connection:
        if not self.product_db_path.exists():
            raise FileNotFoundError(f"product DB not found: {self.product_db_path}")
        uri = f"file:{quote(str(self.product_db_path), safe='/:')}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=15)
        con.row_factory = sqlite3.Row
        return con

    def _fetch_rows(self, sql: str, params: Sequence[Any], required_table: str) -> list[Mapping[str, Any]]:
        if not self.product_db_path.exists():
            return []
        with self.connect_ro() as con:
            if not relation_exists(con, required_table):
                return []
            rows = con.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]


PRODUCT_DB_TABLES = (
    "schema_migrations",
    "tenants",
    "product_calls",
    "job_runs",
    "capture_inbox_items",
    "tenant_manager_owner_map",
    "retention_policies",
)
REQUIRED_PRODUCT_DB_MIGRATIONS = (
    "20260507_001_product_appliance_base",
    "20260507_002_config_history_retention",
    "20260507_003_scheduler_runtime",
    "20260507_004_capture_inbox",
)


def build_product_api_readiness_report(
    product_root: Path,
    product_db_path: Path,
    out_path: Optional[Path] = None,
    workspace_root: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_api_paths(product_root, product_db_path, out_path)
    api = ProductApiFacade(product_root=product_root, product_db_path=product_db_path, workspace_root=workspace_root)
    endpoints = {
        "appliance_dashboard": api.appliance_dashboard(),
        "dashboard_summary": api.dashboard_summary(),
        "capture_recent": api.capture_recent(limit=25),
        "processing_queue": api.processing_queue(limit=25),
        "scheduler_runs": api.scheduler_runs(limit=25),
        "scheduler_health": api.scheduler_health(),
        "scheduler_control_plane": api.scheduler_control_plane(),
        "lifecycle_readiness": api.lifecycle_readiness(),
        "crm_mapping_preview": api.crm_mapping_preview(),
        "asr_gate_status": api.asr_gate_status(),
        "writeback_previews": api.writeback_previews(),
        "operator_runtime_status": api.operator_runtime_status(),
        "manual_resolution_status": api.manual_resolution_status(),
        "waiting_autonomous_work_status": api.waiting_autonomous_work_status(),
        "writeback_dry_run_readiness": api.writeback_dry_run_readiness(),
        "knowledge_playbook": api.knowledge_playbook(),
        "settings_adapters": api.settings_adapters(),
    }
    dashboard_summary = mapping_or_empty(endpoints["dashboard_summary"].get("summary"))
    product_db_present = product_db_path.exists() and product_db_path.is_file()
    readiness_ok = product_db_present and bool(dashboard_summary.get("validation_ok"))
    blocked_reasons = []
    if not product_db_present:
        blocked_reasons.append("product_db_missing")
    if product_db_present and not bool(dashboard_summary.get("validation_ok")):
        blocked_reasons.append("product_db_invalid")
    blocked = max(int(dashboard_summary.get("blocked") or 0), len(blocked_reasons))
    warnings = int(dashboard_summary.get("warnings") or 0)
    report = {
        "summary": ProductApiSummary(
            schema_version=PRODUCT_API_SCHEMA_VERSION,
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            endpoints=len(endpoints),
            read_only=True,
            validation_ok=readiness_ok,
            blocked=blocked,
            warnings=warnings,
        ).to_json_dict()
        | {
            "report_generated_ok": True,
            "product_db_present": product_db_present,
            "blocked_reasons": blocked_reasons,
        },
        "endpoints": endpoints,
        "safety": read_only_actions(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def guard_product_api_paths(product_root: Path, product_db_path: Path, out_path: Optional[Path] = None) -> None:
    for label, path in (("product root", product_root), ("product DB", product_db_path), ("product API audit", out_path)):
        if path is None:
            continue
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
    if not path_is_relative_to(product_db_path, product_root):
        raise ValueError(f"product DB must stay under product root: {product_root}")
    if out_path is not None and not path_is_relative_to(out_path, product_root):
        raise ValueError(f"product API audit must stay under product root: {product_root}")


def read_only_actions() -> Mapping[str, bool]:
    return {
        "read_only": True,
        "downloads_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_transcripts": False,
        "write_runtime_db": False,
        "stable_runtime_writes": False,
        "write_crm": False,
        "write_tallanto": False,
        "dispatch_worker": False,
    }


def writeback_blocked_reasons(preview: Mapping[str, Any]) -> Sequence[str]:
    summary = mapping_or_empty(preview.get("summary"))
    reasons = []
    if int(summary.get("blocked_missing_crm_entity") or 0):
        reasons.append("missing_crm_entity")
    if int(summary.get("blocked_missing_insight") or 0):
        reasons.append("missing_insight_payload")
    if int(summary.get("blocked_policy_forbidden") or 0):
        reasons.append("policy_forbidden_action")
    return reasons


def default_crm_snapshot_path(product_root: Path) -> Optional[Path]:
    for relative in (
        "crm_snapshots/amocrm_entities.json",
        "crm_snapshots/amocrm_entities.jsonl",
        "crm_snapshots/amocrm_entities.csv",
        "config/amocrm_entities.json",
    ):
        path = product_root / relative
        if path.exists() and path.is_file():
            return path
    return None


def bounded_filter_text(value: Any, label: str, limit: int = 120) -> str:
    text = clean(value)
    if len(text) > limit:
        raise ValueError(f"{label} filter is too long")
    return text


def like_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def empty_db_summary(product_db_path: Path) -> Mapping[str, Any]:
    return {
        "schema_version": PRODUCT_API_SCHEMA_VERSION,
        "db_path": str(product_db_path),
        "schema_migrations": 0,
        "tenants": 0,
        "product_calls": 0,
        "job_runs": 0,
        "capture_inbox_items": 0,
        "capture_inbox_ready": 0,
        "capture_inbox_blocked": 0,
        "pending_owner_mappings": 0,
        "retention_policies": 0,
        "validation_ok": False,
        "blocked": 1,
        "blocked_reasons": {
            "missing_relations": list(PRODUCT_DB_TABLES),
            "missing_required_migrations": list(REQUIRED_PRODUCT_DB_MIGRATIONS),
        },
        "warnings": 0,
    }


def count_relation(con: sqlite3.Connection, name: str) -> int:
    if not relation_exists(con, name):
        return 0
    return scalar_int(con, f"select count(*) from {name}")


def relation_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "select 1 from sqlite_master where name = ? and type in ('table', 'view')",
        (clean(name),),
    ).fetchone()
    return row is not None


def scalar_int(con: sqlite3.Connection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def scalar_int_if_relation(con: sqlite3.Connection, relation_name: str, sql: str) -> int:
    if not relation_exists(con, relation_name):
        return 0
    return scalar_int(con, sql)


def missing_required_migrations(con: sqlite3.Connection) -> list[str]:
    if not relation_exists(con, "schema_migrations"):
        return list(REQUIRED_PRODUCT_DB_MIGRATIONS)
    present = {
        clean(row[0])
        for row in con.execute("select migration_id from schema_migrations").fetchall()
    }
    return [migration for migration in REQUIRED_PRODUCT_DB_MIGRATIONS if migration not in present]


def load_json_if_exists(path: Path) -> Optional[Mapping[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, Mapping) else None


def latest_operator_status_path(workspace_root: Path) -> Path:
    stable_runtime = workspace_root / "stable_runtime"
    candidates = [path for path in stable_runtime.glob("operator_status_*/operator_status.json") if path.is_file()]
    if not candidates:
        return stable_runtime / "operator_status_20260511_v1" / "operator_status.json"
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
