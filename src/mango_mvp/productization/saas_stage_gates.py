from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from mango_mvp.productization.test_ingest import clean, path_is_relative_to


SAAS_STAGE_GATES_SCHEMA_VERSION = "saas_stage_gates_v1"
SAAS_STAGE_COUNT = 9
DEFAULT_ENDPOINTS = (
    "GET /dashboard/summary",
    "GET /capture/recent",
    "GET /queues/processing",
    "GET /scheduler/runs",
    "GET /asr/gates",
    "GET /writeback/previews",
    "GET /knowledge/playbook",
    "GET /settings/adapters",
)
DEFAULT_UI_SCREENS = (
    "Dashboard",
    "Capture",
    "Processing Queue",
    "Scheduler Runs",
    "ASR Gates",
    "ROP Queue",
    "AMO Writeback Preview",
    "Knowledge Lab",
    "Settings",
)


@dataclass(frozen=True)
class SaasStageGateSummary:
    schema_version: str
    product_root: str
    product_db_path: str
    stages_total: int
    stages_ready: int
    stages_partial: int
    stages_blocked: int
    stages_planned: int
    saas_ready: bool
    report_generated_ok: bool
    stage_inputs_valid: bool
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SaasStageGate:
    stage: int
    key: str
    title: str
    status: str
    purpose: str
    completed_capabilities: Sequence[str]
    blockers: Sequence[str]
    next_work_packages: Sequence[str]
    safety: Mapping[str, bool]

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_saas_stage_gates_report(
    product_root: Path,
    product_db_path: Path,
    out_path: Optional[Path] = None,
    workspace_root: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    workspace_root = (workspace_root or Path.cwd()).resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_saas_stage_gate_paths(product_root=product_root, product_db_path=product_db_path, out_path=out_path)

    db_audit = safe_product_db_audit(product_db_path=product_db_path, product_root=product_root)
    artifact_state = scan_product_artifacts(product_root=product_root, workspace_root=workspace_root)
    stages = build_stage_gates(db_audit=db_audit, artifact_state=artifact_state)
    status_counts = Counter(stage.status for stage in stages)
    blocked = sum(len(stage.blockers) for stage in stages if stage.status in {"blocked", "planned"})
    warnings = sum(len(stage.blockers) for stage in stages if stage.status == "partial")
    db_summary = mapping_or_empty(db_audit.get("summary"))
    stage_inputs_valid = bool(db_audit.get("present")) and bool(db_summary.get("validation_ok"))
    summary = SaasStageGateSummary(
        schema_version=SAAS_STAGE_GATES_SCHEMA_VERSION,
        product_root=str(product_root),
        product_db_path=str(product_db_path),
        stages_total=SAAS_STAGE_COUNT,
        stages_ready=status_counts.get("ready", 0),
        stages_partial=status_counts.get("partial", 0),
        stages_blocked=status_counts.get("blocked", 0),
        stages_planned=status_counts.get("planned", 0),
        saas_ready=status_counts.get("ready", 0) == SAAS_STAGE_COUNT,
        report_generated_ok=True,
        stage_inputs_valid=stage_inputs_valid,
        validation_ok=stage_inputs_valid,
        blocked=blocked,
        warnings=warnings,
    )
    report = {
        "summary": summary.to_json_dict(),
        "status_counts": dict(sorted(status_counts.items())),
        "stages": [stage.to_json_dict() for stage in stages],
        "stage_index": {stage.key: stage.to_json_dict() for stage in stages},
        "product_db_audit": db_audit,
        "artifact_state": artifact_state,
        "api_surface_contract": build_api_surface_contract(stages),
        "ui_surface_contract": build_ui_surface_contract(stages),
        "writeback_policy": build_writeback_policy(),
        "knowledge_playbook_contract": build_knowledge_playbook_contract(),
        "deployment_profile": build_deployment_profile(product_root=product_root, product_db_path=product_db_path),
        "demo_profile": build_demo_profile(),
        "safety": saas_stage_safety(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def build_stage_gates(db_audit: Mapping[str, Any], artifact_state: Mapping[str, Any]) -> list[SaasStageGate]:
    db_summary = mapping_or_empty(db_audit.get("summary"))
    db_exists = bool(db_audit.get("present"))
    db_ok = bool(db_summary.get("validation_ok")) if db_exists else False
    job_runs = int(db_summary.get("job_runs") or 0)
    capture_ready = int(db_summary.get("capture_inbox_ready") or 0)
    product_calls = int(db_summary.get("product_calls") or 0)
    pending_owner_mappings = int(db_summary.get("pending_owner_mappings") or 0)
    live_poll = bool(artifact_state.get("live_shadow_poll_archive_present"))
    asr_gate = bool(artifact_state.get("asr_stage25_request_present"))
    ui_contract = bool(artifact_state.get("ui_contract_module_present"))
    product_api = bool(artifact_state.get("product_api_module_present"))
    http_api = bool(artifact_state.get("product_api_http_module_present"))
    appliance_loop = bool(artifact_state.get("appliance_loop_module_present"))

    stage1_complete = db_ok and product_api and http_api
    stage1_blockers = []
    if not db_exists:
        stage1_blockers.append("product_db_missing")
    elif not db_ok:
        stage1_blockers.append("product_db_integrity_not_ok")
    if not product_api:
        stage1_blockers.append("product_api_facade_missing")
    if not http_api:
        stage1_blockers.append("product_api_http_layer_missing")
    if stage1_complete:
        stage1_status = "partial"
        stage1_blockers.append("http_server_not_supervised_as_service")
        stage1_blockers.append("single_writer_mutation_endpoints_not_implemented")
    else:
        stage1_status = "blocked"

    stage2_complete = db_ok and job_runs > 0 and capture_ready >= 0 and live_poll and appliance_loop
    stage2_blockers = []
    if not db_ok:
        stage2_blockers.append("product_db_integrity_not_ok")
    if job_runs <= 0:
        stage2_blockers.append("scheduler_jobs_missing")
    if capture_ready <= 0:
        stage2_blockers.append("capture_inbox_ready_empty")
    if not live_poll:
        stage2_blockers.append("live_mango_shadow_poll_archive_missing")
    if not appliance_loop:
        stage2_blockers.append("appliance_loop_dry_run_missing")
    if stage2_complete:
        stage2_status = "partial"
        stage2_blockers.extend(["recording_download_auto_trigger_disabled", "processing_auto_trigger_still_disabled_by_policy"])
    else:
        stage2_status = "blocked"

    stage3_status = "partial" if ui_contract and product_calls > 0 else "planned"
    stage3_blockers = ["frontend_application_not_implemented", "browser_verified_ui_not_available"]
    if product_calls <= 0:
        stage3_blockers.append("product_call_rows_missing_for_ui")

    stage4_status = "partial" if asr_gate and capture_ready > 0 else "planned"
    stage4_blockers = ["asr_execution_approval_record_missing", "worker_launcher_dry_run_not_materialized", "runtime_write_bridge_not_allowed"]
    if not asr_gate:
        stage4_blockers.append("asr_stage25_gate_missing")

    stage5_status = "partial" if pending_owner_mappings == 0 and product_calls > 0 else "planned"
    stage5_blockers = ["crm_writeback_dry_run_diff_not_implemented", "staged_writeback_policy_not_backed_by_queue"]
    if pending_owner_mappings > 0:
        stage5_blockers.append(f"pending_owner_mappings:{pending_owner_mappings}")

    stage6_status = "partial" if bool(artifact_state.get("insight_seed_module_present")) and product_calls > 0 else "planned"
    stage6_blockers = ["client_chain_layer_not_materialized", "outcome_linker_not_materialized", "sales_moment_extractor_not_run"]

    stage7_status = "partial" if db_exists else "planned"
    stage7_blockers = ["installer_or_service_profile_missing", "backup_restore_schedule_not_automated", "secrets_rotation_policy_missing"]

    stage8_status = "planned"
    stage8_blockers = ["demo_tenant_not_materialized", "anonymized_demo_dataset_missing", "demo_script_not_browser_verified"]

    stage9_status = "planned"
    stage9_blockers = ["tenant_isolation_not_enforced", "per_tenant_scheduler_not_materialized", "multi_client_support_runbook_missing"]

    return [
        SaasStageGate(
            stage=1,
            key="local_product_api_single_writer",
            title="Local Product API / single-writer boundary",
            status=stage1_status,
            purpose="Expose product state through one local read-only HTTP/API boundary before any future DB mutations are allowed.",
            completed_capabilities=capabilities(
                ("product DB integrity", db_ok),
                ("Product API facade", product_api),
                ("read-only HTTP route layer", http_api),
                ("8 read-only product routes", product_api and http_api),
            ),
            blockers=stage1_blockers,
            next_work_packages=(
                "Run the HTTP layer under a supervised local service profile.",
                "Add future mutation endpoints only behind explicit policy gates.",
                "Keep SQLite writes single-writer through this backend.",
            ),
            safety=saas_stage_safety(),
        ),
        SaasStageGate(
            stage=2,
            key="internal_autonomous_appliance",
            title="Internal autonomous appliance loop",
            status=stage2_status,
            purpose="Plan Mango poll -> capture inbox -> recording/processing queue as an autonomous appliance loop without executing dangerous actions.",
            completed_capabilities=capabilities(
                ("scheduler runtime rows", job_runs > 0),
                ("capture inbox ready items", capture_ready > 0),
                ("live Mango shadow poll archive", live_poll),
                ("appliance loop dry-run", appliance_loop),
            ),
            blockers=stage2_blockers,
            next_work_packages=(
                "Connect capture inbox -> recording asset -> processing queue as dry-run orchestrator.",
                "Add retry/dead-letter visibility for loop actions.",
                "Keep ASR and CRM disabled until explicit approval gates are complete.",
            ),
            safety=saas_stage_safety(),
        ),
        SaasStageGate(
            stage=3,
            key="ui_v1",
            title="UI v1",
            status=stage3_status,
            purpose="Provide a product dashboard for capture, queues, ROP work, writeback previews and settings.",
            completed_capabilities=capabilities(
                ("dashboard data contract", ui_contract),
                ("product call rows for table views", product_calls > 0),
                ("blocked dangerous UI actions listed", True),
            ),
            blockers=stage3_blockers,
            next_work_packages=(
                "Build read-only UI against API mocks/contracts.",
                "Verify dashboard/capture/queue/settings screens in browser before wiring mutations.",
            ),
            safety=saas_stage_safety(),
        ),
        SaasStageGate(
            stage=4,
            key="processing_orchestration",
            title="Processing orchestration",
            status=stage4_status,
            purpose="Safely connect new captured calls to ASR/processing plans through gates, without running ASR automatically.",
            completed_capabilities=capabilities(
                ("ASR Stage 25 request gate", asr_gate),
                ("capture inbox source rows", capture_ready > 0),
                ("dangerous actions blocked", True),
            ),
            blockers=stage4_blockers,
            next_work_packages=(
                "Create worker launcher dry-run after explicit approval record exists.",
                "Add processing queue rows without writing runtime DB.",
                "Run the first sandbox execution only after separate approval.",
            ),
            safety=saas_stage_safety(),
        ),
        SaasStageGate(
            stage=5,
            key="controlled_crm_writeback",
            title="Controlled CRM writeback",
            status=stage5_status,
            purpose="Move from CRM write prohibition to reviewed previews, staged dry-run diffs and later controlled writes.",
            completed_capabilities=capabilities(
                ("manager owner mapping source", product_calls > 0),
                ("no automatic CRM writes", True),
                ("owner mappings fully confirmed", pending_owner_mappings == 0 and product_calls > 0),
            ),
            blockers=stage5_blockers,
            next_work_packages=(
                "Create CRM writeback preview rows with quality gates.",
                "Resolve pending owner mappings before any staged writeback pilot.",
                "Add staged rollout gates: 50, 300, full.",
            ),
            safety=saas_stage_safety(),
        ),
        SaasStageGate(
            stage=6,
            key="knowledge_sales_playbook",
            title="Knowledge / AI Sales Playbook",
            status=stage6_status,
            purpose="Turn conversations into client-chain, outcome-linked sales moments and ROP-approved playbook items.",
            completed_capabilities=capabilities(
                ("insight seed module", bool(artifact_state.get("insight_seed_module_present"))),
                ("product call corpus for sampling", product_calls > 0),
                ("playbook schema contract", True),
            ),
            blockers=stage6_blockers,
            next_work_packages=(
                "Materialize client chains and opportunity keys.",
                "Add outcome linker over local AMO/Tallanto snapshots.",
                "Run stratified sales moment pilot on 300-500 client chains.",
            ),
            safety=saas_stage_safety(),
        ),
        SaasStageGate(
            stage=7,
            key="client_hosted_packaging",
            title="Client-hosted packaging",
            status=stage7_status,
            purpose="Package the appliance for installation on a client laptop/server with backup, logs, secrets and retention.",
            completed_capabilities=capabilities(
                ("local product root", bool(artifact_state.get("product_root_present"))),
                ("SQLite appliance DB profile", db_exists),
                ("retention policies in DB", int(db_summary.get("retention_policies") or 0) > 0),
            ),
            blockers=stage7_blockers,
            next_work_packages=(
                "Create service launch profile and .env template.",
                "Add scheduled backups and restore drill command.",
                "Document client operator runbook.",
            ),
            safety=saas_stage_safety(),
        ),
        SaasStageGate(
            stage=8,
            key="demo_ready_product",
            title="Demo-ready product",
            status=stage8_status,
            purpose="Show a repeatable client demo without exposing private runtime artifacts.",
            completed_capabilities=capabilities(
                ("demo scenario contract", True),
                ("product dashboard data source", product_calls > 0),
                ("writeback stays preview-only", True),
            ),
            blockers=stage8_blockers,
            next_work_packages=(
                "Create anonymized demo tenant and dataset.",
                "Build scripted demo flow: connect -> capture -> queue -> insight -> CRM preview.",
                "Prepare sales deck/runbook for first client demos.",
            ),
            safety=saas_stage_safety(),
        ),
        SaasStageGate(
            stage=9,
            key="multi_client_readiness",
            title="Multi-client readiness",
            status=stage9_status,
            purpose="Prepare tenant isolation, per-tenant credentials/schedules and support boundaries for more than one client.",
            completed_capabilities=capabilities(
                ("tenant_id is present in product DB model", db_exists),
                ("adapter layer exists", bool(artifact_state.get("adapter_layer_present"))),
                ("single-client product root exists", bool(artifact_state.get("product_root_present"))),
            ),
            blockers=stage9_blockers,
            next_work_packages=(
                "Add tenant isolation audit and per-tenant scheduler config.",
                "Create support/debug bundle per tenant.",
                "Define SQLite single-client vs MariaDB multi-client upgrade profile.",
            ),
            safety=saas_stage_safety(),
        ),
    ]


def build_api_surface_contract(stages: Sequence[SaasStageGate]) -> Mapping[str, Any]:
    return {
        "schema_version": "product_api_surface_contract_v1",
        "mode": "local_read_only_first",
        "endpoints": [
            {
                "path": endpoint,
                "method": endpoint.split(" ", 1)[0],
                "route": endpoint.split(" ", 1)[1],
                "mutates_state": False,
                "requires_policy_gate": False,
            }
            for endpoint in DEFAULT_ENDPOINTS
        ],
        "future_mutations_must_use_gates": [
            "download_audio",
            "run_asr",
            "run_ra",
            "write_crm",
            "write_runtime_db",
        ],
        "stage_status": {stage.key: stage.status for stage in stages},
    }


def build_ui_surface_contract(stages: Sequence[SaasStageGate]) -> Mapping[str, Any]:
    return {
        "schema_version": "product_ui_v1_surface_contract",
        "screens": [
            {
                "screen": screen,
                "data_source": "Product API",
                "direct_runtime_access": False,
            }
            for screen in DEFAULT_UI_SCREENS
        ],
        "blocked_actions": ["download_audio", "run_asr", "run_ra", "write_crm", "write_runtime_db"],
        "stage_status": {stage.key: stage.status for stage in stages},
    }


def build_writeback_policy() -> Mapping[str, Any]:
    return {
        "schema_version": "crm_writeback_policy_v1",
        "current_mode": "preview_only",
        "write_crm": False,
        "write_tallanto": False,
        "required_sequence": ["dry_run_diff", "human_review", "pilot_50", "pilot_300", "full_rollout"],
        "blocking_requirements": [
            "all manager owner mappings confirmed",
            "quality gate passes",
            "rollback/audit report path configured",
            "explicit staged approval recorded",
        ],
    }


def build_knowledge_playbook_contract() -> Mapping[str, Any]:
    return {
        "schema_version": "ai_sales_playbook_contract_v1",
        "entities": [
            "client_identity",
            "opportunity",
            "chain_event",
            "sales_moment",
            "response_quality_score",
            "outcome_link",
            "playbook_item",
        ],
        "first_wave": "300-500 stratified client chains",
        "requires_human_approval_for_bot": True,
        "must_separate": ["observed_correlation", "llm_recommendation", "rop_approved_recommendation"],
    }


def build_deployment_profile(product_root: Path, product_db_path: Path) -> Mapping[str, Any]:
    return {
        "schema_version": "client_hosted_appliance_profile_v1",
        "profile": "sqlite_single_writer_v1",
        "product_root": str(product_root),
        "product_db_path": str(product_db_path),
        "future_db_profiles": ["mariadb"],
        "required_services": ["local_product_api", "scheduler", "worker_supervisor"],
        "required_ops": ["backup", "restore_drill", "log_rotation", "health_check", "retention_audit"],
    }


def build_demo_profile() -> Mapping[str, Any]:
    return {
        "schema_version": "demo_ready_profile_v1",
        "tenant": "demo_anonymized",
        "required_assets": ["demo_product_db", "demo_audio_refs_or_redacted_samples", "demo_playbook_items", "demo_script"],
        "demo_flow": ["connect_mango", "capture_calls", "show_processing_queue", "show_rop_queue", "show_knowledge_item", "show_crm_preview"],
        "privacy_required": True,
    }


def safe_product_db_audit(product_db_path: Path, product_root: Path) -> Mapping[str, Any]:
    if not product_db_path.exists():
        return {"present": False, "summary": {"validation_ok": False}, "error": f"product DB not found: {product_db_path}"}
    try:
        with connect_sqlite_readonly(product_db_path) as con:
            tables = {name: relation_count(con, name) for name in PRODUCT_DB_AUDIT_RELATIONS}
            missing_migrations = missing_required_migrations(con)
            duplicate_event_keys = scalar_int_if_relation(
                con,
                "product_calls",
                """
                SELECT count(*)
                  FROM (
                    SELECT event_key
                      FROM product_calls
                     GROUP BY event_key
                    HAVING count(*) > 1
                  )
                """,
            )
            pending_owner_mappings = scalar_int_if_relation(
                con,
                "tenant_manager_owner_map",
                "select count(*) from tenant_manager_owner_map where decision_status != 'confirmed_candidate'",
            )
            blocked = len(missing_migrations) + duplicate_event_keys
            summary = {
                "schema_version": "product_appliance_sqlite_v1",
                "db_path": str(product_db_path),
                "schema_migrations": tables["schema_migrations"],
                "tenants": tables["tenants"],
                "manager_owner_rows": tables["tenant_manager_owner_map"],
                "product_calls": tables["product_calls"],
                "job_runs": tables["job_runs"],
                "due_job_runs": count_due_job_runs(con),
                "running_job_runs": scalar_int_if_relation(con, "job_runs", "select count(*) from job_runs where status = 'running'"),
                "failed_job_runs": scalar_int_if_relation(con, "job_runs", "select count(*) from job_runs where status = 'failed'"),
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
                "calls_with_crm_owner": scalar_int_if_relation(
                    con,
                    "product_calls",
                    "select count(*) from product_calls where crm_owner_id is not null",
                ),
                "pending_owner_mappings": pending_owner_mappings,
                "raw_payload_refs_present": scalar_int_if_relation(
                    con,
                    "product_calls",
                    "select count(*) from product_calls where raw_payload_ref is not null and raw_payload_ref != ''",
                ),
                "job_types": tables["job_types"],
                "tenant_config_history": tables["tenant_config_history"],
                "retention_policies": tables["retention_policies"],
                "validation_ok": blocked == 0,
                "blocked": blocked,
                "warnings": pending_owner_mappings,
            }
        return {
            "present": True,
            "summary": summary,
            "tables": tables,
            "missing_migrations": missing_migrations,
            "blocked_reasons": {
                "missing_required_migrations": len(missing_migrations),
                "duplicate_event_keys": duplicate_event_keys,
            },
            "warning_reasons": {"pending_owner_mappings": pending_owner_mappings},
            "read_only": True,
        }
    except Exception as exc:
        return {"present": True, "summary": {"validation_ok": False}, "error": str(exc)}


def scan_product_artifacts(product_root: Path, workspace_root: Path) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    workspace_root = workspace_root.resolve(strict=False)
    return {
        "product_root_present": product_root.exists() and product_root.is_dir(),
        "live_shadow_poll_archive_present": any_file_under(
            product_root / "raw_payload_archive" / "live_shadow_poll",
            patterns=("*.jsonl",),
        )
        or any_file_under(
            product_root / "scheduler_outputs",
            patterns=("shadow_poll_job_*.json",),
        ),
        "asr_stage25_request_present": (
            product_root
            / "asr_worker_sandbox_execution_request_stage25"
            / "asr_worker_sandbox_execution_request_stage25.json"
        ).exists(),
        "ui_contract_module_present": (workspace_root / "src" / "mango_mvp" / "productization" / "ui_contracts.py").exists(),
        "insight_seed_module_present": (workspace_root / "src" / "mango_mvp" / "productization" / "insight_seed.py").exists(),
        "product_api_module_present": (workspace_root / "src" / "mango_mvp" / "productization" / "product_api.py").exists(),
        "product_api_http_module_present": (workspace_root / "src" / "mango_mvp" / "productization" / "product_api_http.py").exists(),
        "appliance_loop_module_present": (workspace_root / "src" / "mango_mvp" / "productization" / "appliance_loop.py").exists(),
        "adapter_layer_present": (workspace_root / "src" / "mango_mvp" / "productization" / "adapters.py").exists(),
        "stage25_artifact_dir_present": (product_root / "asr_worker_sandbox_execution_request_stage25").exists(),
    }


def any_existing(*paths: Path) -> bool:
    return any(path.exists() for path in paths)


def any_file_under(path: Path, patterns: Sequence[str]) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for pattern in patterns:
        if any(candidate.is_file() for candidate in path.glob(pattern)):
            return True
    return False


def capabilities(*items: tuple[str, bool]) -> list[str]:
    return [name for name, enabled in items if enabled]


def saas_stage_safety() -> Mapping[str, bool]:
    return {
        "read_only_inputs": True,
        "writes_report_json": True,
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


def guard_saas_stage_gate_paths(product_root: Path, product_db_path: Path, out_path: Optional[Path]) -> None:
    for label, path in (("product root", product_root), ("product DB", product_db_path), ("SaaS stage gate output", out_path)):
        if path is None:
            continue
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
    if not path_is_relative_to(product_db_path, product_root):
        raise ValueError(f"product DB must stay under product root: {product_root}")
    if out_path is not None and not path_is_relative_to(out_path, product_root):
        raise ValueError(f"SaaS stage gate output must stay under product root: {product_root}")


PRODUCT_DB_AUDIT_RELATIONS = (
    "schema_migrations",
    "tenants",
    "tenant_manager_owner_map",
    "product_calls",
    "job_types",
    "job_runs",
    "capture_inbox_items",
    "tenant_config_history",
    "retention_policies",
)

REQUIRED_PRODUCT_DB_MIGRATIONS = (
    "20260507_001_product_appliance_base",
    "20260507_002_config_history_retention",
    "20260507_003_scheduler_runtime",
    "20260507_004_capture_inbox",
)


def connect_sqlite_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(db_path), safe='/:')}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=15)
    con.row_factory = sqlite3.Row
    return con


def relation_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "select 1 from sqlite_master where name = ? and type in ('table', 'view')",
        (clean(name),),
    ).fetchone()
    return row is not None


def relation_count(con: sqlite3.Connection, name: str) -> int:
    if not relation_exists(con, name):
        return 0
    return scalar_int(con, f"select count(*) from {name}")


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


def count_due_job_runs(con: sqlite3.Connection) -> int:
    if not relation_exists(con, "job_runs"):
        return 0
    return scalar_int(
        con,
        """
        SELECT count(*)
          FROM job_runs
         WHERE status in ('planned', 'retry_wait')
           AND (next_run_at IS NULL OR next_run_at <= datetime('now'))
        """,
    )


def mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
