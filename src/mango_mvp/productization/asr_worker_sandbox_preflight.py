from __future__ import annotations

import importlib.util
import json
import os
import shutil
from collections import Counter
from collections.abc import Callable, Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.asr_execution_approval_gate import ASR_JOB_TYPE
from mango_mvp.productization.asr_scheduler_dry_run import DANGEROUS_HARD_GUARDS, load_json_object, mapping_or_empty, optional_int
from mango_mvp.productization.asr_worker_sandbox_execution_contract import (
    ASR_WORKER_SANDBOX_EXECUTION_CONTRACT_SCHEMA_VERSION,
)
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_WORKER_SANDBOX_PREFLIGHT_SCHEMA_VERSION = "asr_worker_sandbox_preflight_v1"
DEFAULT_DISK_SAFETY_MARGIN_BYTES = 64 * 1024 * 1024
ENGINE_MODULES = {
    "mlx": "mlx_whisper",
    "gigaam": "gigaam",
    "openai": "openai",
}


@dataclass(frozen=True)
class AsrWorkerSandboxPreflightSummary:
    schema_version: str
    product_root: str
    contract_path: str
    out_dir: str
    preflight_report_path: str
    contract_sha256: str
    preflight_report_sha256: str
    approval_ref: Optional[str]
    selected_engine: Optional[str]
    tasks: int
    passed_tasks: int
    blocked_tasks: int
    audio_files_checked: int
    audio_sha_ok: int
    output_collisions: int
    dir_preflight_ok: bool
    disk_space_ok: bool
    engine_preflight_ok: bool
    required_free_bytes: int
    available_free_bytes: int
    dispatch_allowed: bool
    run_asr: bool
    write_transcripts: bool
    validation_ok: bool
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_asr_worker_sandbox_preflight(
    product_root: Path,
    contract_path: Path,
    out_dir: Path,
    preflight_report_path: Path,
    out_path: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    module_checker: Optional[Callable[[str], bool]] = None,
    binary_checker: Optional[Callable[[str], Optional[str]]] = None,
    disk_usage_provider: Optional[Callable[[Path], Any]] = None,
    disk_safety_margin_bytes: int = DEFAULT_DISK_SAFETY_MARGIN_BYTES,
) -> Mapping[str, Any]:
    paths = resolve_preflight_paths(
        product_root=product_root,
        contract_path=contract_path,
        out_dir=out_dir,
        preflight_report_path=preflight_report_path,
        out_path=out_path,
    )
    product_root = paths["product_root"]
    contract_path = paths["contract_path"]
    out_dir = paths["out_dir"]
    preflight_report_path = paths["preflight_report_path"]
    out_path = paths.get("out_path")

    contract = load_json_object(contract_path)
    contract_sha = sha256_file(contract_path)
    source_reasons = validate_contract_for_preflight(contract)
    selected_engine = clean(contract.get("selected_engine")).lower()
    sandbox_roots = mapping_or_empty(contract.get("sandbox_roots"))
    output_root = Path(clean(sandbox_roots.get("output_root"))).resolve(strict=False) if clean(sandbox_roots.get("output_root")) else out_dir / "sandbox_outputs"
    tmp_root = Path(clean(sandbox_roots.get("tmp_root"))).resolve(strict=False) if clean(sandbox_roots.get("tmp_root")) else out_dir / "sandbox_tmp"
    root_reasons = validate_sandbox_roots(product_root=product_root, output_root=output_root, tmp_root=tmp_root)
    tasks = contract.get("tasks") if isinstance(contract.get("tasks"), SequenceABC) and not isinstance(contract.get("tasks"), (str, bytes)) else []
    preflight_tasks: list[Mapping[str, Any]] = []
    if not source_reasons and not root_reasons:
        preflight_tasks = [
            build_task_preflight(
                task=task,
                row_number=index,
                product_root=product_root,
                output_root=output_root,
                tmp_root=tmp_root,
            )
            for index, task in enumerate(tasks, start=1)
        ]
        preflight_tasks = apply_cross_task_preflight_blocks(preflight_tasks)
    engine_preflight = build_engine_preflight(
        selected_engine=selected_engine,
        contract=contract,
        env=env or os.environ,
        module_checker=module_checker or module_available,
        binary_checker=binary_checker or shutil.which,
    )
    disk_preflight = build_disk_preflight(
        out_dir=out_dir,
        contract=contract,
        disk_usage_provider=disk_usage_provider or shutil.disk_usage,
        disk_safety_margin_bytes=disk_safety_margin_bytes,
    )
    directory_preflight = build_directory_preflight(output_root=output_root, tmp_root=tmp_root)
    actions = build_preflight_actions(
        source_reasons=sorted(set(source_reasons + root_reasons)),
        task_preflights=preflight_tasks,
        engine_preflight=engine_preflight,
        disk_preflight=disk_preflight,
        directory_preflight=directory_preflight,
    )
    preflight_report = build_preflight_payload(
        contract_path=contract_path,
        contract=contract,
        contract_sha256=contract_sha,
        source_reasons=sorted(set(source_reasons + root_reasons)),
        task_preflights=preflight_tasks,
        engine_preflight=engine_preflight,
        disk_preflight=disk_preflight,
        directory_preflight=directory_preflight,
        actions=actions,
    )
    write_json(preflight_report_path, preflight_report)
    preflight_sha = sha256_file(preflight_report_path)
    workload = mapping_or_empty(preflight_report.get("workload"))
    validation_ok = bool(preflight_report.get("preflight_ready"))
    summary = AsrWorkerSandboxPreflightSummary(
        schema_version=ASR_WORKER_SANDBOX_PREFLIGHT_SCHEMA_VERSION,
        product_root=str(product_root),
        contract_path=str(contract_path),
        out_dir=str(out_dir),
        preflight_report_path=str(preflight_report_path),
        contract_sha256=contract_sha,
        preflight_report_sha256=preflight_sha,
        approval_ref=clean(contract.get("approval_ref")) or None,
        selected_engine=selected_engine or None,
        tasks=optional_int(workload.get("tasks")) or 0,
        passed_tasks=optional_int(workload.get("passed_tasks")) or 0,
        blocked_tasks=optional_int(workload.get("blocked_tasks")) or 0,
        audio_files_checked=optional_int(workload.get("audio_files_checked")) or 0,
        audio_sha_ok=optional_int(workload.get("audio_sha_ok")) or 0,
        output_collisions=optional_int(workload.get("output_collisions")) or 0,
        dir_preflight_ok=bool(directory_preflight.get("ok")),
        disk_space_ok=bool(disk_preflight.get("ok")),
        engine_preflight_ok=bool(engine_preflight.get("ok")),
        required_free_bytes=optional_int(disk_preflight.get("required_free_bytes")) or 0,
        available_free_bytes=optional_int(disk_preflight.get("available_free_bytes")) or 0,
        dispatch_allowed=False,
        run_asr=False,
        write_transcripts=False,
        validation_ok=validation_ok,
        warnings=0 if validation_ok else 1,
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": action_counts_for(preflight_tasks if preflight_report.get("preflight_ready") else actions),
        "actions": actions,
        "preflight_report": preflight_report,
        "task_samples": preflight_tasks[:20],
        "source_contract": {
            "schema_version": clean(contract.get("schema_version")) or None,
            "status": clean(contract.get("status")) or None,
            "approval_ref": clean(contract.get("approval_ref")) or None,
            "selected_engine": selected_engine or None,
            "sha256": contract_sha,
        },
        "safety": sandbox_preflight_safety(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def validate_contract_for_preflight(contract: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if clean(contract.get("schema_version")) != ASR_WORKER_SANDBOX_EXECUTION_CONTRACT_SCHEMA_VERSION:
        reasons.append("contract_schema_unexpected")
    if clean(contract.get("job_type")) != ASR_JOB_TYPE:
        reasons.append("contract_job_type_unexpected")
    if clean(contract.get("mode")) != "sandbox_execution_contract_dry_run":
        reasons.append("contract_mode_unexpected")
    if clean(contract.get("status")) != "sandbox_execution_contract_planned_not_dispatched":
        reasons.append("contract_status_unexpected")
    if not clean(contract.get("approval_ref")):
        reasons.append("contract_approval_ref_missing")
    if not clean(contract.get("selected_engine")):
        reasons.append("contract_selected_engine_missing")
    for key in ("dispatch_allowed", "run_asr", "execution_allowed", "write_outputs", "write_transcripts", "write_runtime_db", "write_crm"):
        if bool(contract.get(key)):
            reasons.append(f"contract_{key}_must_be_false")
    sandbox_roots = mapping_or_empty(contract.get("sandbox_roots"))
    if bool(sandbox_roots.get("write_outputs_in_this_stage")):
        reasons.append("contract_sandbox_roots_must_not_write_outputs")
    if bool(sandbox_roots.get("create_dirs_in_this_stage")):
        reasons.append("contract_sandbox_roots_must_not_create_dirs")
    hard_guards = mapping_or_empty(contract.get("hard_guards"))
    if not hard_guards:
        reasons.append("contract_hard_guards_missing")
    for guard in tuple(DANGEROUS_HARD_GUARDS) + ("dispatch_worker", "load_asr_model", "write_transcripts"):
        if bool(hard_guards.get(guard)):
            reasons.append(f"contract_hard_guard_{guard}_must_be_false")
    next_stage = mapping_or_empty(contract.get("next_stage_contract"))
    if not bool(next_stage.get("may_request_explicit_asr_execution_approval")):
        reasons.append("contract_next_stage_execution_approval_not_allowed")
    if bool(next_stage.get("may_dispatch_worker_in_this_stage")):
        reasons.append("contract_next_stage_must_not_dispatch_worker")
    if bool(next_stage.get("may_run_asr_in_this_stage")):
        reasons.append("contract_next_stage_must_not_run_asr")
    for key in (
        "must_run_final_preflight_before_execution",
        "must_verify_audio_sha256_before_execution",
        "must_use_sandbox_output_root",
        "must_not_write_runtime_db",
        "must_not_write_crm",
        "must_not_touch_stable_runtime",
    ):
        if not bool(next_stage.get(key)):
            reasons.append(f"contract_next_stage_{key}_missing")
    workload = mapping_or_empty(contract.get("workload"))
    planned = optional_int(workload.get("planned_tasks")) or 0
    source_items = optional_int(workload.get("source_items")) or 0
    blocked = optional_int(workload.get("blocked_tasks")) or 0
    if planned <= 0:
        reasons.append("contract_planned_tasks_empty")
    if planned != source_items:
        reasons.append("contract_planned_tasks_do_not_match_source_items")
    if blocked:
        reasons.append("contract_has_blocked_tasks")
    tasks = contract.get("tasks")
    if not isinstance(tasks, SequenceABC) or isinstance(tasks, (str, bytes)):
        reasons.append("contract_tasks_missing")
    elif len(tasks) != planned:
        reasons.append("contract_task_count_mismatch")
    return sorted(set(reasons))


def validate_sandbox_roots(product_root: Path, output_root: Path, tmp_root: Path) -> list[str]:
    reasons: list[str] = []
    for label, path in (("output_root", output_root), ("tmp_root", tmp_root)):
        if "stable_runtime" in path.parts:
            reasons.append(f"sandbox_{label}_under_stable_runtime")
        if not path_is_relative_to(path, product_root):
            reasons.append(f"sandbox_{label}_outside_product_root")
    if output_root == tmp_root:
        reasons.append("sandbox_output_and_tmp_roots_must_differ")
    return sorted(set(reasons))


def build_task_preflight(
    task: Mapping[str, Any],
    row_number: int,
    product_root: Path,
    output_root: Path,
    tmp_root: Path,
) -> Mapping[str, Any]:
    blocked: list[str] = []
    warnings: list[str] = []
    if not isinstance(task, MappingABC):
        return {
            "action": "BLOCK_ASR_SANDBOX_PREFLIGHT_TASK",
            "row_number": row_number,
            "blocked_reasons": ["task_not_object"],
            "warnings": [],
        }
    audio = mapping_or_empty(task.get("audio"))
    sandbox_paths = mapping_or_empty(task.get("sandbox_paths"))
    command = mapping_or_empty(task.get("worker_command_contract"))
    expected_sha = clean(audio.get("sha256")).lower()
    audio_path = Path(clean(audio.get("path"))).resolve(strict=False) if clean(audio.get("path")) else None
    actual_sha: Optional[str] = None
    actual_size: Optional[int] = None
    if clean(task.get("action")) != "PLAN_ASR_SANDBOX_TASK":
        blocked.append("task_action_unexpected")
    if command.get("executable") is not None:
        blocked.append("task_command_executable_must_be_null")
    argv = command.get("argv")
    if not isinstance(argv, SequenceABC) or isinstance(argv, (str, bytes)) or len(argv) != 0:
        blocked.append("task_command_argv_must_be_empty")
    for key in ("dry_run_only",):
        if not bool(command.get(key)):
            blocked.append(f"task_command_{key}_must_be_true")
    for key in ("dispatch_allowed", "run_asr", "write_outputs", "write_transcripts", "write_runtime_db", "write_crm"):
        if bool(command.get(key)):
            blocked.append(f"task_command_{key}_must_be_false")
    if audio_path is None:
        blocked.append("audio_path_missing")
    elif "stable_runtime" in audio_path.parts:
        blocked.append("audio_under_stable_runtime")
    elif not path_is_relative_to(audio_path, product_root):
        blocked.append("audio_outside_product_root")
    elif not audio_path.exists():
        blocked.append("audio_missing")
    elif not audio_path.is_file():
        blocked.append("audio_not_file")
    else:
        actual_size = audio_path.stat().st_size
        expected_size = optional_int(audio.get("size_bytes")) or 0
        if actual_size <= 0:
            blocked.append("audio_zero_size")
        if actual_size != expected_size:
            blocked.append("audio_size_mismatch")
        if expected_sha:
            actual_sha = sha256_file(audio_path)
            if actual_sha != expected_sha:
                blocked.append("audio_sha256_mismatch")
        else:
            blocked.append("audio_sha256_missing")
    output_paths = normalized_sandbox_output_paths(sandbox_paths)
    if not output_paths:
        blocked.append("sandbox_output_paths_missing")
    for key, path in output_paths.items():
        if "stable_runtime" in path.parts:
            blocked.append(f"sandbox_path_{key}_under_stable_runtime")
        allowed_root = tmp_root if key == "task_tmp_dir" else output_root
        if not path_is_relative_to(path, allowed_root):
            blocked.append(f"sandbox_path_{key}_outside_allowed_root")
        if key != "task_tmp_dir" and path.exists():
            blocked.append(f"output_collision:{key}")
        if key == "task_tmp_dir" and path.exists():
            warnings.append("task_tmp_dir_already_exists")
    action = "PASS_ASR_SANDBOX_PREFLIGHT_TASK" if not blocked else "BLOCK_ASR_SANDBOX_PREFLIGHT_TASK"
    return {
        "action": action,
        "reason": "all_preflight_checks_passed" if not blocked else ",".join(sorted(set(blocked))),
        "row_number": row_number,
        "queue_item_id": clean(task.get("queue_item_id")) or None,
        "asset_id": optional_int(task.get("asset_id")),
        "selected_engine": clean(task.get("selected_engine")) or None,
        "audio": {
            "path": str(audio_path) if audio_path is not None else None,
            "expected_sha256": expected_sha or None,
            "actual_sha256": actual_sha,
            "sha256_ok": bool(actual_sha and expected_sha and actual_sha == expected_sha),
            "expected_size_bytes": optional_int(audio.get("size_bytes")) or 0,
            "actual_size_bytes": actual_size,
            "size_ok": actual_size == (optional_int(audio.get("size_bytes")) or 0) if actual_size is not None else False,
        },
        "sandbox_paths": {key: str(value) for key, value in output_paths.items()},
        "blocked_reasons": sorted(set(blocked)),
        "warnings": sorted(set(warnings)),
        "dispatch_allowed": False,
        "run_asr": False,
        "write_transcripts": False,
    }


def normalized_sandbox_output_paths(sandbox_paths: Mapping[str, Any]) -> Mapping[str, Path]:
    result: dict[str, Path] = {}
    for key in ("task_tmp_dir", "transcript_json", "transcript_txt", "asr_audit_json", "engine_stdout_log", "engine_stderr_log"):
        value = clean(sandbox_paths.get(key))
        if value:
            result[key] = Path(value).resolve(strict=False)
    return result


def apply_cross_task_preflight_blocks(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    path_counts = Counter(
        clean(value)
        for item in items
        for key, value in mapping_or_empty(item.get("sandbox_paths")).items()
        if key != "task_tmp_dir" and clean(value)
    )
    queue_counts = Counter(clean(item.get("queue_item_id")) for item in items if clean(item.get("queue_item_id")))
    result: list[Mapping[str, Any]] = []
    for source in items:
        item = dict(source)
        blocked = list(item.get("blocked_reasons") or [])
        if queue_counts.get(clean(item.get("queue_item_id")), 0) > 1:
            blocked.append("duplicate_queue_item_id")
        duplicate_paths = [
            clean(value)
            for key, value in mapping_or_empty(item.get("sandbox_paths")).items()
            if key != "task_tmp_dir" and clean(value) and path_counts.get(clean(value), 0) > 1
        ]
        if duplicate_paths:
            blocked.append("duplicate_sandbox_output_path")
        if blocked:
            item["blocked_reasons"] = sorted(set(blocked))
            item["action"] = "BLOCK_ASR_SANDBOX_PREFLIGHT_TASK"
            item["reason"] = ",".join(item["blocked_reasons"])
        result.append(item)
    return result


def build_engine_preflight(
    selected_engine: str,
    contract: Mapping[str, Any],
    env: Mapping[str, str],
    module_checker: Callable[[str], bool],
    binary_checker: Callable[[str], Optional[str]],
) -> Mapping[str, Any]:
    module_name = ENGINE_MODULES.get(selected_engine)
    blocked: list[str] = []
    if not selected_engine:
        blocked.append("selected_engine_missing")
    if not module_name:
        blocked.append(f"selected_engine_unknown:{selected_engine}")
    module_ok = bool(module_name and module_checker(module_name))
    if module_name and not module_ok:
        blocked.append(f"missing_python_module:{module_name}")
    ffmpeg_path: Optional[str] = None
    if selected_engine == "gigaam":
        ffmpeg_path = binary_checker("ffmpeg")
        if not ffmpeg_path:
            blocked.append("missing_binary:ffmpeg")
    if selected_engine == "openai" and not clean(env.get("OPENAI_API_KEY")):
        blocked.append("missing_env:OPENAI_API_KEY")
    candidate = mapping_or_empty(mapping_or_empty(contract.get("engine_selection")).get("candidate"))
    if candidate and not bool(candidate.get("ready")):
        blocked.append("contract_engine_candidate_not_ready")
    return {
        "ok": not blocked,
        "selected_engine": selected_engine or None,
        "module": module_name,
        "module_available": module_ok,
        "ffmpeg_path": ffmpeg_path,
        "openai_api_key_present": bool(clean(env.get("OPENAI_API_KEY"))) if selected_engine == "openai" else None,
        "checks_import_specs_only": True,
        "loads_models": False,
        "runs_asr": False,
        "blocked_reasons": sorted(set(blocked)),
    }


def build_disk_preflight(
    out_dir: Path,
    contract: Mapping[str, Any],
    disk_usage_provider: Callable[[Path], Any],
    disk_safety_margin_bytes: int,
) -> Mapping[str, Any]:
    resource = mapping_or_empty(contract.get("batch_resource_limits"))
    estimated_tmp = optional_int(resource.get("estimated_tmp_bytes")) or 0
    total_audio = optional_int(resource.get("total_audio_bytes")) or 0
    required = max(0, estimated_tmp + total_audio + int(disk_safety_margin_bytes))
    disk_path = out_dir.parent if out_dir.parent.exists() else nearest_existing_parent(out_dir) or out_dir
    usage = disk_usage_provider(disk_path)
    free = int(getattr(usage, "free", usage[2] if isinstance(usage, tuple) and len(usage) >= 3 else 0))
    free_bucket = (free // DEFAULT_DISK_SAFETY_MARGIN_BYTES) * DEFAULT_DISK_SAFETY_MARGIN_BYTES
    return {
        "ok": free_bucket >= required,
        "out_dir": str(out_dir),
        "disk_path_checked": str(disk_path),
        "required_free_bytes": required,
        "available_free_bytes": free_bucket,
        "available_free_bytes_bucket_size": DEFAULT_DISK_SAFETY_MARGIN_BYTES,
        "estimated_tmp_bytes": estimated_tmp,
        "total_audio_bytes": total_audio,
        "safety_margin_bytes": int(disk_safety_margin_bytes),
        "blocked_reasons": [] if free_bucket >= required else ["insufficient_disk_space"],
    }


def build_directory_preflight(output_root: Path, tmp_root: Path) -> Mapping[str, Any]:
    output_parent = nearest_existing_parent(output_root)
    tmp_parent = nearest_existing_parent(tmp_root)
    output_ok = bool(output_parent and os.access(output_parent, os.W_OK | os.X_OK))
    tmp_ok = bool(tmp_parent and os.access(tmp_parent, os.W_OK | os.X_OK))
    blocked = []
    if not output_ok:
        blocked.append("sandbox_output_root_parent_not_writable")
    if not tmp_ok:
        blocked.append("sandbox_tmp_root_parent_not_writable")
    return {
        "ok": not blocked,
        "creates_dirs": False,
        "output_root": str(output_root),
        "tmp_root": str(tmp_root),
        "output_root_exists": output_root.exists(),
        "tmp_root_exists": tmp_root.exists(),
        "output_parent_checked": str(output_parent) if output_parent else None,
        "tmp_parent_checked": str(tmp_parent) if tmp_parent else None,
        "output_parent_writable": output_ok,
        "tmp_parent_writable": tmp_ok,
        "blocked_reasons": blocked,
    }


def nearest_existing_parent(path: Path) -> Optional[Path]:
    current = path.resolve(strict=False)
    while not current.exists():
        if current.parent == current:
            return None
        current = current.parent
    return current if current.is_dir() else current.parent


def build_preflight_actions(
    source_reasons: Sequence[str],
    task_preflights: Sequence[Mapping[str, Any]],
    engine_preflight: Mapping[str, Any],
    disk_preflight: Mapping[str, Any],
    directory_preflight: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if source_reasons:
        return [
            {
                "action": "BLOCK_ASR_SANDBOX_PREFLIGHT_CONTRACT",
                "reason": "source_contract_failed_preflight_contract",
                "technical_reasons": list(source_reasons),
                "dispatch_allowed": False,
                "run_asr": False,
            }
        ]
    blocked_tasks = [item for item in task_preflights if item.get("action") == "BLOCK_ASR_SANDBOX_PREFLIGHT_TASK"]
    blocked_reasons = []
    if blocked_tasks:
        blocked_reasons.append("blocked_preflight_tasks")
    for source in (engine_preflight, disk_preflight, directory_preflight):
        blocked_reasons.extend(source.get("blocked_reasons") or [])
    if blocked_reasons:
        return [
            {
                "action": "BLOCK_ASR_SANDBOX_FINAL_PREFLIGHT",
                "reason": "one_or_more_final_preflight_checks_failed",
                "technical_reasons": sorted(set(blocked_reasons)),
                "blocked_tasks": len(blocked_tasks),
                "dispatch_allowed": False,
                "run_asr": False,
            }
        ]
    return [
        {
            "action": "PASS_ASR_SANDBOX_FINAL_PREFLIGHT",
            "reason": "sandbox_contract_passed_final_preflight_without_execution",
            "task_count": len(task_preflights),
            "dispatch_allowed": False,
            "run_asr": False,
        }
    ]


def build_preflight_payload(
    contract_path: Path,
    contract: Mapping[str, Any],
    contract_sha256: str,
    source_reasons: Sequence[str],
    task_preflights: Sequence[Mapping[str, Any]],
    engine_preflight: Mapping[str, Any],
    disk_preflight: Mapping[str, Any],
    directory_preflight: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    blocked_tasks = sum(1 for item in task_preflights if item.get("action") == "BLOCK_ASR_SANDBOX_PREFLIGHT_TASK")
    passed_tasks = sum(1 for item in task_preflights if item.get("action") == "PASS_ASR_SANDBOX_PREFLIGHT_TASK")
    output_collisions = sum(
        1
        for item in task_preflights
        for reason in item.get("blocked_reasons") or []
        if clean(reason).startswith("output_collision:")
    )
    audio_sha_ok = sum(1 for item in task_preflights if bool(mapping_or_empty(item.get("audio")).get("sha256_ok")))
    preflight_ready = (
        not source_reasons
        and passed_tasks > 0
        and blocked_tasks == 0
        and bool(engine_preflight.get("ok"))
        and bool(disk_preflight.get("ok"))
        and bool(directory_preflight.get("ok"))
    )
    return {
        "schema_version": ASR_WORKER_SANDBOX_PREFLIGHT_SCHEMA_VERSION,
        "job_type": ASR_JOB_TYPE,
        "mode": "sandbox_final_preflight_dry_run",
        "status": "preflight_passed_not_dispatched" if preflight_ready else "blocked_sandbox_final_preflight",
        "approval_ref": clean(contract.get("approval_ref")) or None,
        "selected_engine": clean(contract.get("selected_engine")) or None,
        "preflight_ready": preflight_ready,
        "dispatch_allowed": False,
        "run_asr": False,
        "execution_allowed": False,
        "write_outputs": False,
        "write_transcripts": False,
        "write_runtime_db": False,
        "write_crm": False,
        "input_refs": {
            "contract_path": str(contract_path),
            "contract_sha256": contract_sha256,
            "readiness_report_path": mapping_or_empty(contract.get("input_refs")).get("readiness_report_path"),
            "readiness_report_sha256": mapping_or_empty(contract.get("input_refs")).get("readiness_report_sha256"),
            "worker_plan_path": mapping_or_empty(contract.get("input_refs")).get("worker_plan_path"),
            "worker_plan_sha256": mapping_or_empty(contract.get("input_refs")).get("worker_plan_sha256"),
        },
        "workload": {
            "tasks": len(task_preflights),
            "passed_tasks": passed_tasks,
            "blocked_tasks": blocked_tasks,
            "audio_files_checked": len([item for item in task_preflights if mapping_or_empty(item.get("audio")).get("actual_sha256")]),
            "audio_sha_ok": audio_sha_ok,
            "output_collisions": output_collisions,
        },
        "engine_preflight": engine_preflight,
        "disk_preflight": disk_preflight,
        "directory_preflight": directory_preflight,
        "source_validation_reasons": list(source_reasons),
        "actions": list(actions),
        "task_preflights": list(task_preflights),
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
            "may_request_explicit_asr_execution_approval": preflight_ready,
            "may_run_asr_in_this_stage": False,
            "may_dispatch_worker_in_this_stage": False,
            "must_keep_sandbox_output_root": True,
            "must_not_write_runtime_db": True,
            "must_not_write_crm": True,
            "must_not_touch_stable_runtime": True,
        },
    }


def resolve_preflight_paths(
    product_root: Path,
    contract_path: Path,
    out_dir: Path,
    preflight_report_path: Path,
    out_path: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "contract_path": contract_path.resolve(strict=False),
        "out_dir": out_dir.resolve(strict=False),
        "preflight_report_path": preflight_report_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    guard_preflight_paths(**paths)
    return paths


def guard_preflight_paths(
    product_root: Path,
    contract_path: Path,
    out_dir: Path,
    preflight_report_path: Path,
    out_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("ASR sandbox contract", contract_path),
        ("ASR sandbox preflight output directory", out_dir),
        ("ASR sandbox preflight report", preflight_report_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not contract_path.exists() or not contract_path.is_file():
        raise FileNotFoundError(f"ASR sandbox contract not found: {contract_path}")
    if not path_is_relative_to(preflight_report_path, out_dir):
        raise ValueError(f"ASR sandbox preflight report must stay under output directory: {out_dir}")
    if out_path is not None:
        if "stable_runtime" in out_path.parts:
            raise ValueError("refusing ASR sandbox preflight audit under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"ASR sandbox preflight audit must stay under product root: {product_root}")
        if not path_is_relative_to(out_path, out_dir):
            raise ValueError(f"ASR sandbox preflight audit must stay under output directory: {out_dir}")


def sandbox_preflight_safety() -> Mapping[str, bool]:
    return {
        "read_only_inputs": True,
        "reads_contract": True,
        "reads_audio_for_sha256": True,
        "writes_preflight_report": True,
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


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def action_counts_for(items: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return dict(sorted(Counter(clean(item.get("action")) for item in items).items()))
