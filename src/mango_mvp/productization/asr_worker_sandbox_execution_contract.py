from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.asr_execution_approval_gate import ASR_JOB_TYPE
from mango_mvp.productization.asr_scheduler_dry_run import DANGEROUS_HARD_GUARDS, load_json_object, mapping_or_empty, optional_int
from mango_mvp.productization.asr_worker_execution_dry_run import ASR_WORKER_EXECUTION_DRY_RUN_SCHEMA_VERSION
from mango_mvp.productization.asr_worker_sandbox_readiness import (
    ASR_WORKER_SANDBOX_READINESS_SCHEMA_VERSION,
    validate_worker_plan_for_sandbox,
)
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_WORKER_SANDBOX_EXECUTION_CONTRACT_SCHEMA_VERSION = "asr_worker_sandbox_execution_contract_v1"
SANDBOX_TASK_CONTRACT_VERSION = "asr_worker_sandbox_task_v1"
DEFAULT_ENGINE_PREFERENCE = ("mlx", "gigaam", "openai")


@dataclass(frozen=True)
class AsrWorkerSandboxExecutionContractSummary:
    schema_version: str
    product_root: str
    readiness_report_path: str
    worker_plan_path: str
    out_dir: str
    contract_path: str
    readiness_report_sha256: str
    worker_plan_sha256: str
    approval_ref: Optional[str]
    selected_engine: Optional[str]
    engine_selection_reason: str
    source_items: int
    tasks: int
    blocked_tasks: int
    total_duration_sec: float
    total_audio_bytes: int
    estimated_tmp_bytes: int
    estimated_timeout_sec: int
    sandbox_output_root: str
    sandbox_tmp_root: str
    dispatch_allowed: bool
    run_asr: bool
    write_transcripts: bool
    validation_ok: bool
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_asr_worker_sandbox_execution_contract(
    product_root: Path,
    readiness_report_path: Path,
    out_dir: Path,
    contract_path: Path,
    out_path: Optional[Path] = None,
    worker_plan_path: Optional[Path] = None,
    preferred_engine: Optional[str] = None,
    sandbox_output_root: Optional[Path] = None,
    sandbox_tmp_root: Optional[Path] = None,
) -> Mapping[str, Any]:
    paths = resolve_sandbox_contract_paths(
        product_root=product_root,
        readiness_report_path=readiness_report_path,
        out_dir=out_dir,
        contract_path=contract_path,
        out_path=out_path,
        worker_plan_path=worker_plan_path,
        sandbox_output_root=sandbox_output_root,
        sandbox_tmp_root=sandbox_tmp_root,
    )
    product_root = paths["product_root"]
    readiness_report_path = paths["readiness_report_path"]
    out_dir = paths["out_dir"]
    contract_path = paths["contract_path"]
    out_path = paths.get("out_path")

    readiness_report = load_json_object(readiness_report_path)
    worker_plan_path = paths.get("worker_plan_path") or infer_worker_plan_path(
        readiness_report=readiness_report,
        product_root=product_root,
    )
    sandbox_output_root = paths.get("sandbox_output_root") or out_dir / "sandbox_outputs"
    sandbox_tmp_root = paths.get("sandbox_tmp_root") or out_dir / "sandbox_tmp"
    guard_inferred_sandbox_contract_paths(
        product_root=product_root,
        out_dir=out_dir,
        worker_plan_path=worker_plan_path,
        sandbox_output_root=sandbox_output_root,
        sandbox_tmp_root=sandbox_tmp_root,
    )

    worker_plan = load_json_object(worker_plan_path)
    readiness_sha = sha256_file(readiness_report_path)
    worker_plan_sha = sha256_file(worker_plan_path)
    readiness_reasons = validate_readiness_for_sandbox_contract(readiness_report, worker_plan_path, worker_plan_sha)
    worker_reasons = validate_worker_plan_for_sandbox(worker_plan)
    engine_choice = select_asr_engine(readiness_report, preferred_engine=preferred_engine)
    source_reasons = sorted(set(readiness_reasons + [f"worker_plan:{reason}" for reason in worker_reasons] + engine_choice["blocked_reasons"]))

    tasks: list[Mapping[str, Any]] = []
    worker_envelopes = worker_plan.get("worker_envelopes")
    if not source_reasons and isinstance(worker_envelopes, SequenceABC) and not isinstance(worker_envelopes, (str, bytes)):
        tasks = [
            build_sandbox_task(
                envelope=envelope,
                row_number=index,
                selected_engine=clean(engine_choice.get("selected_engine")) or "",
                sandbox_output_root=sandbox_output_root,
                sandbox_tmp_root=sandbox_tmp_root,
            )
            for index, envelope in enumerate(worker_envelopes, start=1)
        ]
        tasks = apply_duplicate_task_blocks(tasks)
    actions = build_contract_actions(source_reasons=source_reasons, tasks=tasks, engine_choice=engine_choice)
    contract = build_contract_payload(
        readiness_report_path=readiness_report_path,
        readiness_report=readiness_report,
        readiness_report_sha256=readiness_sha,
        worker_plan_path=worker_plan_path,
        worker_plan=worker_plan,
        worker_plan_sha256=worker_plan_sha,
        source_reasons=source_reasons,
        engine_choice=engine_choice,
        tasks=tasks,
        sandbox_output_root=sandbox_output_root,
        sandbox_tmp_root=sandbox_tmp_root,
        actions=actions,
    )
    write_json(contract_path, contract)
    contract_sha = sha256_file(contract_path)
    workload = mapping_or_empty(contract.get("workload"))
    resource_limits = mapping_or_empty(contract.get("batch_resource_limits"))
    selected_engine = clean(engine_choice.get("selected_engine")) or None
    blocked_tasks = optional_int(workload.get("blocked_tasks")) or 0
    planned_tasks = optional_int(workload.get("planned_tasks")) or 0
    validation_ok = not source_reasons and planned_tasks > 0 and blocked_tasks == 0
    summary = AsrWorkerSandboxExecutionContractSummary(
        schema_version=ASR_WORKER_SANDBOX_EXECUTION_CONTRACT_SCHEMA_VERSION,
        product_root=str(product_root),
        readiness_report_path=str(readiness_report_path),
        worker_plan_path=str(worker_plan_path),
        out_dir=str(out_dir),
        contract_path=str(contract_path),
        readiness_report_sha256=readiness_sha,
        worker_plan_sha256=worker_plan_sha,
        approval_ref=clean(readiness_report.get("approval_ref")) or None,
        selected_engine=selected_engine,
        engine_selection_reason=clean(engine_choice.get("selection_reason")) or "not_selected",
        source_items=optional_int(workload.get("source_items")) or 0,
        tasks=planned_tasks,
        blocked_tasks=blocked_tasks,
        total_duration_sec=float(resource_limits.get("total_duration_sec") or 0.0),
        total_audio_bytes=optional_int(resource_limits.get("total_audio_bytes")) or 0,
        estimated_tmp_bytes=optional_int(resource_limits.get("estimated_tmp_bytes")) or 0,
        estimated_timeout_sec=optional_int(resource_limits.get("estimated_timeout_sec")) or 0,
        sandbox_output_root=str(sandbox_output_root),
        sandbox_tmp_root=str(sandbox_tmp_root),
        dispatch_allowed=False,
        run_asr=False,
        write_transcripts=False,
        validation_ok=validation_ok,
        warnings=count_task_warnings(tasks) + (0 if validation_ok else 1),
    )
    report = {
        "summary": {
            **summary.to_json_dict(),
            "contract_sha256": contract_sha,
        },
        "action_counts": action_counts_for(actions if source_reasons else tasks),
        "actions": actions,
        "contract": contract,
        "task_samples": tasks[:20],
        "source_readiness_report": {
            "schema_version": clean(readiness_report.get("schema_version")) or None,
            "status": clean(readiness_report.get("status")) or None,
            "approval_ref": clean(readiness_report.get("approval_ref")) or None,
            "sha256": readiness_sha,
        },
        "source_worker_plan": {
            "schema_version": clean(worker_plan.get("schema_version")) or None,
            "status": clean(worker_plan.get("status")) or None,
            "approval_ref": clean(worker_plan.get("approval_ref")) or None,
            "sha256": worker_plan_sha,
        },
        "safety": sandbox_contract_safety(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def validate_readiness_for_sandbox_contract(
    readiness_report: Mapping[str, Any],
    worker_plan_path: Path,
    worker_plan_sha256: str,
) -> list[str]:
    reasons: list[str] = []
    if clean(readiness_report.get("schema_version")) != ASR_WORKER_SANDBOX_READINESS_SCHEMA_VERSION:
        reasons.append("readiness_schema_unexpected")
    if clean(readiness_report.get("job_type")) != ASR_JOB_TYPE:
        reasons.append("readiness_job_type_unexpected")
    if clean(readiness_report.get("mode")) != "worker_sandbox_readiness_gate":
        reasons.append("readiness_mode_unexpected")
    if clean(readiness_report.get("status")) != "sandbox_ready_dry_run":
        reasons.append("readiness_status_not_ready")
    for key in ("worker_sandbox_ready", "asr_engine_ready"):
        if not bool(readiness_report.get(key)):
            reasons.append(f"readiness_{key}_must_be_true")
    for key in ("dispatch_allowed", "run_asr", "write_transcripts", "write_runtime_db", "write_crm"):
        if bool(readiness_report.get(key)):
            reasons.append(f"readiness_{key}_must_be_false")
    hard_guards = mapping_or_empty(readiness_report.get("hard_guards"))
    if not hard_guards:
        reasons.append("readiness_hard_guards_missing")
    for guard in tuple(DANGEROUS_HARD_GUARDS) + ("dispatch_worker", "load_asr_model", "write_transcripts"):
        if bool(hard_guards.get(guard)):
            reasons.append(f"readiness_hard_guard_{guard}_must_be_false")
    next_stage = mapping_or_empty(readiness_report.get("next_stage_contract"))
    if not bool(next_stage.get("may_create_worker_sandbox_execution_plan")):
        reasons.append("readiness_next_stage_contract_not_allowed")
    if bool(next_stage.get("may_dispatch_worker_in_this_stage")):
        reasons.append("readiness_next_stage_must_not_dispatch_worker")
    if bool(next_stage.get("may_run_asr_in_this_stage")):
        reasons.append("readiness_next_stage_must_not_run_asr")
    if not bool(next_stage.get("must_select_engine_before_execution")):
        reasons.append("readiness_engine_selection_requirement_missing")
    if not bool(next_stage.get("must_require_explicit_execution_approval")):
        reasons.append("readiness_explicit_execution_approval_requirement_missing")
    input_refs = mapping_or_empty(readiness_report.get("input_refs"))
    readiness_worker_path = clean(input_refs.get("worker_plan_path"))
    if not readiness_worker_path:
        reasons.append("readiness_worker_plan_path_missing")
    elif Path(readiness_worker_path).resolve(strict=False) != worker_plan_path.resolve(strict=False):
        reasons.append("readiness_worker_plan_path_mismatch")
    if clean(input_refs.get("worker_plan_sha256")) != worker_plan_sha256:
        reasons.append("readiness_worker_plan_sha_mismatch")
    workload = mapping_or_empty(readiness_report.get("workload"))
    source_items = optional_int(workload.get("source_items")) or 0
    planned = optional_int(workload.get("planned_envelopes")) or 0
    blocked = optional_int(workload.get("blocked_envelopes")) or 0
    if source_items <= 0:
        reasons.append("readiness_source_items_empty")
    if planned != source_items:
        reasons.append("readiness_planned_envelopes_do_not_match_source_items")
    if blocked:
        reasons.append("readiness_has_blocked_envelopes")
    capability = mapping_or_empty(readiness_report.get("capability_report"))
    if not bool(capability.get("asr_engine_ready")):
        reasons.append("readiness_capability_asr_engine_not_ready")
    return sorted(set(reasons))


def select_asr_engine(readiness_report: Mapping[str, Any], preferred_engine: Optional[str]) -> Mapping[str, Any]:
    requested = clean(preferred_engine).lower()
    if requested == "auto":
        requested = ""
    capability = mapping_or_empty(readiness_report.get("capability_report"))
    candidates = [
        candidate
        for candidate in capability.get("engine_candidates", [])
        if isinstance(candidate, MappingABC)
    ]
    ready_real = [
        candidate
        for candidate in candidates
        if bool(candidate.get("ready")) and bool(candidate.get("counts_as_real_asr"))
    ]
    ready_names = [clean(candidate.get("engine")).lower() for candidate in ready_real if clean(candidate.get("engine"))]
    if requested:
        if requested in ready_names:
            selected = next(candidate for candidate in ready_real if clean(candidate.get("engine")).lower() == requested)
            return {
                "selected_engine": requested,
                "selection_reason": "preferred_engine_ready",
                "preferred_engine": requested,
                "ready_real_engines": ready_names,
                "candidate": selected,
                "blocked_reasons": [],
            }
        return {
            "selected_engine": None,
            "selection_reason": "preferred_engine_not_ready",
            "preferred_engine": requested,
            "ready_real_engines": ready_names,
            "candidate": None,
            "blocked_reasons": [f"engine_not_ready:{requested}"],
        }
    for engine in DEFAULT_ENGINE_PREFERENCE:
        if engine in ready_names:
            selected = next(candidate for candidate in ready_real if clean(candidate.get("engine")).lower() == engine)
            return {
                "selected_engine": engine,
                "selection_reason": "auto_preference_order",
                "preferred_engine": None,
                "ready_real_engines": ready_names,
                "candidate": selected,
                "blocked_reasons": [],
            }
    return {
        "selected_engine": None,
        "selection_reason": "no_ready_real_engine",
        "preferred_engine": requested or None,
        "ready_real_engines": ready_names,
        "candidate": None,
        "blocked_reasons": ["no_ready_real_engine"],
    }


def build_sandbox_task(
    envelope: Mapping[str, Any],
    row_number: int,
    selected_engine: str,
    sandbox_output_root: Path,
    sandbox_tmp_root: Path,
) -> Mapping[str, Any]:
    queue_item_id = clean(envelope.get("queue_item_id"))
    queue_short = queue_item_id[:16] if queue_item_id else f"row_{row_number:06d}"
    tenant_id = clean(envelope.get("tenant_id")) or "unknown_tenant"
    provider = clean(envelope.get("provider")) or "unknown_provider"
    audio = mapping_or_empty(envelope.get("audio"))
    resource = mapping_or_empty(envelope.get("resource_estimate"))
    output_dir = sandbox_output_root / selected_engine / tenant_id / provider
    tmp_dir = sandbox_tmp_root / selected_engine / queue_short
    stem = f"{row_number:06d}__{queue_short}"
    task: dict[str, Any] = {
        "action": "PLAN_ASR_SANDBOX_TASK",
        "reason": "worker_envelope_mapped_to_sandbox_execution_contract_without_execution",
        "row_number": row_number,
        "contract_version": SANDBOX_TASK_CONTRACT_VERSION,
        "queue_item_id": queue_item_id or None,
        "asset_id": optional_int(envelope.get("asset_id")),
        "tenant_id": tenant_id,
        "provider": provider,
        "event_key": clean(envelope.get("event_key")) or None,
        "recording_id": clean(envelope.get("recording_id")) or None,
        "selected_engine": selected_engine,
        "audio": {
            "path": clean(audio.get("path")) or None,
            "rel_path": clean(audio.get("rel_path")) or None,
            "sha256": clean(audio.get("sha256")).lower() or None,
            "size_bytes": optional_int(audio.get("size_bytes")) or 0,
            "duration_sec": optional_float(audio.get("duration_sec")) or 0.0,
            "preflight_must_verify_sha256_before_execution": True,
        },
        "source_outputs": dict(mapping_or_empty(envelope.get("outputs"))),
        "sandbox_paths": {
            "task_tmp_dir": str(tmp_dir),
            "transcript_json": str(output_dir / f"{stem}.transcript.json"),
            "transcript_txt": str(output_dir / f"{stem}.transcript.txt"),
            "asr_audit_json": str(output_dir / f"{stem}.asr_audit.json"),
            "engine_stdout_log": str(output_dir / f"{stem}.stdout.log"),
            "engine_stderr_log": str(output_dir / f"{stem}.stderr.log"),
        },
        "resource_limits": {
            "timeout_sec": optional_int(resource.get("estimated_timeout_sec")) or 60,
            "max_tmp_bytes": optional_int(resource.get("estimated_tmp_bytes")) or 0,
            "max_audio_bytes": optional_int(resource.get("audio_bytes")) or 0,
            "max_duration_sec": optional_float(resource.get("audio_duration_sec")) or 0.0,
            "estimated_cpu_seconds_min": optional_float(resource.get("estimated_cpu_seconds_min")) or 0.0,
            "estimated_cpu_seconds_max": optional_float(resource.get("estimated_cpu_seconds_max")) or 0.0,
            "size_class": clean(resource.get("size_class")) or "unknown",
        },
        "worker_command_contract": {
            "executable": None,
            "argv": [],
            "dry_run_only": True,
            "dispatch_allowed": False,
            "run_asr": False,
            "write_outputs": False,
            "write_transcripts": False,
            "write_runtime_db": False,
            "write_crm": False,
            "selected_engine": selected_engine,
            "adapter_contract": "future_asr_worker_sandbox_adapter",
        },
        "preflight_checks": {
            "audio_file_exists": False,
            "audio_file_size_matches_plan": False,
            "sandbox_paths_under_output_root": False,
            "no_stable_runtime_paths": False,
        },
        "blocked_reasons": [],
        "warnings": [],
    }
    blocked = list(task["blocked_reasons"])
    audio_path = Path(clean(audio.get("path"))).resolve(strict=False) if clean(audio.get("path")) else None
    if clean(envelope.get("action")) != "PLAN_ASR_WORKER_ENVELOPE":
        blocked.append("source_envelope_not_planned")
    if not selected_engine:
        blocked.append("selected_engine_missing")
    for field in ("queue_item_id", "event_key", "recording_id"):
        if not clean(task.get(field)):
            blocked.append(f"missing_{field}")
    if audio_path is None:
        blocked.append("audio_path_missing")
    elif "stable_runtime" in audio_path.parts:
        blocked.append("audio_under_stable_runtime")
    elif not audio_path.exists():
        blocked.append("audio_missing")
    elif not audio_path.is_file():
        blocked.append("audio_not_file")
    else:
        actual_size = audio_path.stat().st_size
        expected_size = optional_int(audio.get("size_bytes")) or 0
        task["preflight_checks"]["audio_file_exists"] = True
        task["preflight_checks"]["audio_file_size_matches_plan"] = actual_size == expected_size
        if actual_size != expected_size:
            blocked.append("audio_size_mismatch")
    sandbox_paths = [Path(clean(value)) for value in mapping_or_empty(task["sandbox_paths"]).values() if clean(value)]
    task["preflight_checks"]["sandbox_paths_under_output_root"] = all(
        path_is_relative_to(path.resolve(strict=False), sandbox_output_root.resolve(strict=False))
        or path_is_relative_to(path.resolve(strict=False), sandbox_tmp_root.resolve(strict=False))
        for path in sandbox_paths
    )
    task["preflight_checks"]["no_stable_runtime_paths"] = all("stable_runtime" not in path.parts for path in sandbox_paths)
    if not task["preflight_checks"]["sandbox_paths_under_output_root"]:
        blocked.append("sandbox_path_outside_allowed_roots")
    if not task["preflight_checks"]["no_stable_runtime_paths"]:
        blocked.append("sandbox_path_under_stable_runtime")
    for value in mapping_or_empty(task["source_outputs"]).values():
        if clean(value) and "stable_runtime" in Path(clean(value)).parts:
            blocked.append("source_output_under_stable_runtime")
    if blocked:
        task["blocked_reasons"] = sorted(set(blocked))
        task["action"] = "BLOCK_ASR_SANDBOX_TASK"
        task["reason"] = ",".join(task["blocked_reasons"])
    return task


def apply_duplicate_task_blocks(tasks: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    queue_counts = Counter(clean(item.get("queue_item_id")) for item in tasks if clean(item.get("queue_item_id")))
    path_counts = Counter(
        clean(value)
        for item in tasks
        for value in mapping_or_empty(item.get("sandbox_paths")).values()
        if clean(value)
    )
    result: list[Mapping[str, Any]] = []
    for source in tasks:
        item = dict(source)
        blocked = list(item.get("blocked_reasons") or [])
        if item["action"] == "PLAN_ASR_SANDBOX_TASK":
            if queue_counts.get(clean(item.get("queue_item_id")), 0) > 1:
                blocked.append("duplicate_queue_item_id")
            if any(path_counts.get(clean(value), 0) > 1 for value in mapping_or_empty(item.get("sandbox_paths")).values() if clean(value)):
                blocked.append("duplicate_sandbox_path")
            if blocked:
                item["blocked_reasons"] = sorted(set(blocked))
                item["action"] = "BLOCK_ASR_SANDBOX_TASK"
                item["reason"] = ",".join(item["blocked_reasons"])
        result.append(item)
    return result


def build_contract_actions(
    source_reasons: Sequence[str],
    tasks: Sequence[Mapping[str, Any]],
    engine_choice: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if source_reasons:
        return [
            {
                "action": "BLOCK_ASR_SANDBOX_EXECUTION_CONTRACT",
                "reason": "source_readiness_or_engine_contract_failed",
                "technical_reasons": list(source_reasons),
                "selected_engine": clean(engine_choice.get("selected_engine")) or None,
                "dispatch_allowed": False,
                "run_asr": False,
            }
        ]
    blocked = [item for item in tasks if item.get("action") == "BLOCK_ASR_SANDBOX_TASK"]
    if blocked:
        return [
            {
                "action": "BLOCK_ASR_SANDBOX_EXECUTION_CONTRACT_TASKS",
                "reason": "one_or_more_sandbox_tasks_blocked",
                "blocked_tasks": len(blocked),
                "selected_engine": clean(engine_choice.get("selected_engine")) or None,
                "dispatch_allowed": False,
                "run_asr": False,
            }
        ]
    return [
        {
            "action": "PLAN_ASR_SANDBOX_EXECUTION_CONTRACT",
            "reason": "sandbox_execution_contract_planned_without_dispatch_or_asr",
            "task_count": len(tasks),
            "selected_engine": clean(engine_choice.get("selected_engine")) or None,
            "dispatch_allowed": False,
            "run_asr": False,
        }
    ]


def build_contract_payload(
    readiness_report_path: Path,
    readiness_report: Mapping[str, Any],
    readiness_report_sha256: str,
    worker_plan_path: Path,
    worker_plan: Mapping[str, Any],
    worker_plan_sha256: str,
    source_reasons: Sequence[str],
    engine_choice: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    sandbox_output_root: Path,
    sandbox_tmp_root: Path,
    actions: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    blocked_tasks = sum(1 for item in tasks if item.get("action") == "BLOCK_ASR_SANDBOX_TASK")
    planned_tasks = sum(1 for item in tasks if item.get("action") == "PLAN_ASR_SANDBOX_TASK")
    source_ok = not source_reasons
    tasks_ok = source_ok and planned_tasks > 0 and blocked_tasks == 0
    return {
        "schema_version": ASR_WORKER_SANDBOX_EXECUTION_CONTRACT_SCHEMA_VERSION,
        "job_type": ASR_JOB_TYPE,
        "mode": "sandbox_execution_contract_dry_run",
        "status": "sandbox_execution_contract_planned_not_dispatched" if tasks_ok else "blocked_sandbox_execution_contract",
        "approval_ref": clean(readiness_report.get("approval_ref")) or clean(worker_plan.get("approval_ref")) or None,
        "selected_engine": clean(engine_choice.get("selected_engine")) or None,
        "engine_selection": engine_choice,
        "dispatch_allowed": False,
        "run_asr": False,
        "execution_allowed": False,
        "write_outputs": False,
        "write_transcripts": False,
        "write_runtime_db": False,
        "write_crm": False,
        "input_refs": {
            "readiness_report_path": str(readiness_report_path),
            "readiness_report_sha256": readiness_report_sha256,
            "worker_plan_path": str(worker_plan_path),
            "worker_plan_sha256": worker_plan_sha256,
            "execution_plan_path": mapping_or_empty(readiness_report.get("input_refs")).get("execution_plan_path"),
            "execution_plan_sha256": mapping_or_empty(readiness_report.get("input_refs")).get("execution_plan_sha256"),
            "pack_manifest_path": mapping_or_empty(readiness_report.get("input_refs")).get("pack_manifest_path"),
            "pack_manifest_sha256": mapping_or_empty(readiness_report.get("input_refs")).get("pack_manifest_sha256"),
        },
        "sandbox_roots": {
            "output_root": str(sandbox_output_root),
            "tmp_root": str(sandbox_tmp_root),
            "write_outputs_in_this_stage": False,
            "create_dirs_in_this_stage": False,
        },
        "workload": {
            "source_items": len(tasks),
            "planned_tasks": planned_tasks,
            "blocked_tasks": blocked_tasks,
        },
        "batch_resource_limits": aggregate_task_resource_limits(tasks),
        "actions": list(actions),
        "tasks": list(tasks),
        "source_validation_reasons": list(source_reasons),
        "hard_guards": {
            "dispatch_worker": False,
            "load_asr_model": False,
            "run_asr": False,
            "run_ra": False,
            "write_transcripts": False,
            "write_runtime_db": False,
            "write_product_db": False,
            "write_asset_db": False,
            "write_crm": False,
            "write_tallanto": False,
            "touch_stable_runtime": False,
        },
        "next_stage_contract": {
            "may_request_explicit_asr_execution_approval": tasks_ok,
            "may_run_asr_in_this_stage": False,
            "may_dispatch_worker_in_this_stage": False,
            "must_run_final_preflight_before_execution": True,
            "must_verify_audio_sha256_before_execution": True,
            "must_use_sandbox_output_root": True,
            "must_not_write_runtime_db": True,
            "must_not_write_crm": True,
            "must_not_touch_stable_runtime": True,
        },
    }


def aggregate_task_resource_limits(tasks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    planned = [item for item in tasks if item.get("action") == "PLAN_ASR_SANDBOX_TASK"]
    limits = [mapping_or_empty(item.get("resource_limits")) for item in planned]
    size_counts = Counter(clean(limit.get("size_class")) for limit in limits if clean(limit.get("size_class")))
    return {
        "planned_tasks": len(planned),
        "total_duration_sec": round(sum(float(limit.get("max_duration_sec") or 0.0) for limit in limits), 3),
        "total_audio_bytes": sum(int(limit.get("max_audio_bytes") or 0) for limit in limits),
        "estimated_tmp_bytes": sum(int(limit.get("max_tmp_bytes") or 0) for limit in limits),
        "estimated_timeout_sec": sum(int(limit.get("timeout_sec") or 0) for limit in limits),
        "max_single_timeout_sec": max((int(limit.get("timeout_sec") or 0) for limit in limits), default=0),
        "estimated_cpu_seconds_min": round(sum(float(limit.get("estimated_cpu_seconds_min") or 0.0) for limit in limits), 3),
        "estimated_cpu_seconds_max": round(sum(float(limit.get("estimated_cpu_seconds_max") or 0.0) for limit in limits), 3),
        "size_class_counts": dict(sorted(size_counts.items())),
    }


def resolve_sandbox_contract_paths(
    product_root: Path,
    readiness_report_path: Path,
    out_dir: Path,
    contract_path: Path,
    out_path: Optional[Path],
    worker_plan_path: Optional[Path],
    sandbox_output_root: Optional[Path],
    sandbox_tmp_root: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "readiness_report_path": readiness_report_path.resolve(strict=False),
        "out_dir": out_dir.resolve(strict=False),
        "contract_path": contract_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    if worker_plan_path is not None:
        paths["worker_plan_path"] = worker_plan_path.resolve(strict=False)
    if sandbox_output_root is not None:
        paths["sandbox_output_root"] = sandbox_output_root.resolve(strict=False)
    if sandbox_tmp_root is not None:
        paths["sandbox_tmp_root"] = sandbox_tmp_root.resolve(strict=False)
    guard_sandbox_contract_paths(**paths)
    return paths


def guard_sandbox_contract_paths(
    product_root: Path,
    readiness_report_path: Path,
    out_dir: Path,
    contract_path: Path,
    out_path: Optional[Path] = None,
    worker_plan_path: Optional[Path] = None,
    sandbox_output_root: Optional[Path] = None,
    sandbox_tmp_root: Optional[Path] = None,
) -> None:
    for label, path in (
        ("ASR sandbox readiness report", readiness_report_path),
        ("ASR sandbox contract output directory", out_dir),
        ("ASR sandbox execution contract", contract_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not readiness_report_path.exists() or not readiness_report_path.is_file():
        raise FileNotFoundError(f"ASR sandbox readiness report not found: {readiness_report_path}")
    if not path_is_relative_to(contract_path, out_dir):
        raise ValueError(f"ASR sandbox execution contract must stay under output directory: {out_dir}")
    for label, path in (
        ("ASR worker plan", worker_plan_path),
        ("ASR sandbox output root", sandbox_output_root),
        ("ASR sandbox tmp root", sandbox_tmp_root),
        ("ASR sandbox audit", out_path),
    ):
        if path is None:
            continue
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if out_path is not None and not path_is_relative_to(out_path, out_dir):
        raise ValueError(f"ASR sandbox audit must stay under output directory: {out_dir}")


def guard_inferred_sandbox_contract_paths(
    product_root: Path,
    out_dir: Path,
    worker_plan_path: Path,
    sandbox_output_root: Path,
    sandbox_tmp_root: Path,
) -> None:
    for label, path in (
        ("ASR worker plan", worker_plan_path),
        ("ASR sandbox output root", sandbox_output_root),
        ("ASR sandbox tmp root", sandbox_tmp_root),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not worker_plan_path.exists() or not worker_plan_path.is_file():
        raise FileNotFoundError(f"ASR worker plan not found: {worker_plan_path}")
    if not path_is_relative_to(sandbox_output_root, out_dir):
        raise ValueError(f"ASR sandbox output root must stay under output directory: {out_dir}")
    if not path_is_relative_to(sandbox_tmp_root, out_dir):
        raise ValueError(f"ASR sandbox tmp root must stay under output directory: {out_dir}")


def infer_worker_plan_path(readiness_report: Mapping[str, Any], product_root: Path) -> Path:
    raw = clean(mapping_or_empty(readiness_report.get("input_refs")).get("worker_plan_path"))
    if not raw:
        raise ValueError("ASR worker plan path is missing from readiness report input_refs")
    path = Path(raw).resolve(strict=False)
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"ASR worker plan must stay under product root: {product_root}")
    return path


def sandbox_contract_safety() -> Mapping[str, bool]:
    return {
        "read_only_inputs": True,
        "reads_readiness_report": True,
        "reads_worker_plan": True,
        "writes_execution_contract": True,
        "creates_sandbox_output_dirs": False,
        "creates_sandbox_tmp_dirs": False,
        "imports_asr_modules": False,
        "loads_models": False,
        "dispatch_worker": False,
        "downloads_audio": False,
        "copies_audio": False,
        "hardlinks_audio": False,
        "product_db_writes": False,
        "asset_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "run_asr": False,
        "run_ra": False,
        "write_transcripts": False,
        "write_crm": False,
        "write_tallanto": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def action_counts_for(items: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return dict(sorted(Counter(clean(item.get("action")) for item in items).items()))


def count_task_warnings(tasks: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(item.get("warnings") or []) for item in tasks)


def optional_float(value: Any) -> Optional[float]:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
