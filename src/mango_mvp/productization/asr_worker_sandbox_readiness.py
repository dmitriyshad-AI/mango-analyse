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
from mango_mvp.productization.asr_worker_execution_dry_run import ASR_WORKER_EXECUTION_DRY_RUN_SCHEMA_VERSION
from mango_mvp.productization.recording_asset_ingest import sha256_file
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


ASR_WORKER_SANDBOX_READINESS_SCHEMA_VERSION = "asr_worker_sandbox_readiness_v1"

ASR_ENV_REFS = (
    "TRANSCRIBE_PROVIDER",
    "DUAL_TRANSCRIBE_ENABLED",
    "SECONDARY_TRANSCRIBE_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_TRANSCRIBE_MODEL",
    "MLX_WHISPER_MODEL",
    "GIGAAM_MODEL",
    "GIGAAM_DEVICE",
    "GIGAAM_SEGMENT_SEC",
    "TRANSCRIBE_LANGUAGE",
)


@dataclass(frozen=True)
class AsrWorkerSandboxReadinessSummary:
    schema_version: str
    product_root: str
    worker_plan_path: str
    out_dir: str
    readiness_report_path: str
    worker_plan_sha256: str
    readiness_report_sha256: str
    approval_ref: Optional[str]
    source_plan_valid: bool
    worker_sandbox_ready: bool
    asr_engine_ready: bool
    ready_real_engines: int
    engine_candidates: int
    source_items: int
    envelopes: int
    blocked_envelopes: int
    total_duration_sec: float
    total_audio_bytes: int
    dispatch_allowed: bool
    run_asr: bool
    validation_ok: bool
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_asr_worker_sandbox_readiness(
    product_root: Path,
    worker_plan_path: Path,
    out_dir: Path,
    readiness_report_path: Path,
    out_path: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    module_checker: Optional[Callable[[str], bool]] = None,
    binary_checker: Optional[Callable[[str], Optional[str]]] = None,
) -> Mapping[str, Any]:
    paths = resolve_sandbox_readiness_paths(
        product_root=product_root,
        worker_plan_path=worker_plan_path,
        out_dir=out_dir,
        readiness_report_path=readiness_report_path,
        out_path=out_path,
    )
    product_root = paths["product_root"]
    worker_plan_path = paths["worker_plan_path"]
    out_dir = paths["out_dir"]
    readiness_report_path = paths["readiness_report_path"]
    out_path = paths.get("out_path")

    worker_plan = load_json_object(worker_plan_path)
    worker_plan_sha = sha256_file(worker_plan_path)
    source_reasons = validate_worker_plan_for_sandbox(worker_plan)
    capability_report = build_capability_report(
        env=env or os.environ,
        module_checker=module_checker or module_available,
        binary_checker=binary_checker or shutil.which,
    )
    ready_real_engines = [
        engine
        for engine in capability_report["engine_candidates"]
        if bool(engine.get("ready")) and bool(engine.get("counts_as_real_asr"))
    ]
    source_plan_valid = not source_reasons
    asr_engine_ready = bool(ready_real_engines)
    worker_sandbox_ready = source_plan_valid and asr_engine_ready
    actions = build_readiness_actions(
        source_reasons=source_reasons,
        asr_engine_ready=asr_engine_ready,
        ready_real_engines=ready_real_engines,
    )
    readiness_report = build_readiness_payload(
        worker_plan_path=worker_plan_path,
        worker_plan=worker_plan,
        worker_plan_sha256=worker_plan_sha,
        source_reasons=source_reasons,
        capability_report=capability_report,
        ready_real_engines=ready_real_engines,
        actions=actions,
    )
    write_json(readiness_report_path, readiness_report)
    readiness_sha = sha256_file(readiness_report_path)
    workload = mapping_or_empty(worker_plan.get("workload"))
    resources = mapping_or_empty(worker_plan.get("resource_estimate"))
    summary = AsrWorkerSandboxReadinessSummary(
        schema_version=ASR_WORKER_SANDBOX_READINESS_SCHEMA_VERSION,
        product_root=str(product_root),
        worker_plan_path=str(worker_plan_path),
        out_dir=str(out_dir),
        readiness_report_path=str(readiness_report_path),
        worker_plan_sha256=worker_plan_sha,
        readiness_report_sha256=readiness_sha,
        approval_ref=clean(worker_plan.get("approval_ref")) or None,
        source_plan_valid=source_plan_valid,
        worker_sandbox_ready=worker_sandbox_ready,
        asr_engine_ready=asr_engine_ready,
        ready_real_engines=len(ready_real_engines),
        engine_candidates=len(capability_report["engine_candidates"]),
        source_items=optional_int(workload.get("source_items")) or 0,
        envelopes=optional_int(workload.get("envelopes")) or 0,
        blocked_envelopes=optional_int(workload.get("blocked_envelopes")) or 0,
        total_duration_sec=float(resources.get("total_duration_sec") or 0.0),
        total_audio_bytes=optional_int(resources.get("total_audio_bytes")) or 0,
        dispatch_allowed=False,
        run_asr=False,
        validation_ok=source_plan_valid,
        warnings=0 if worker_sandbox_ready else 1,
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": action_counts_for(actions),
        "actions": actions,
        "readiness_report": readiness_report,
        "capability_report": capability_report,
        "source_worker_plan": {
            "schema_version": clean(worker_plan.get("schema_version")) or None,
            "job_type": clean(worker_plan.get("job_type")) or None,
            "mode": clean(worker_plan.get("mode")) or None,
            "status": clean(worker_plan.get("status")) or None,
            "approval_ref": clean(worker_plan.get("approval_ref")) or None,
            "sha256": worker_plan_sha,
        },
        "safety": sandbox_readiness_safety(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def validate_worker_plan_for_sandbox(worker_plan: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if clean(worker_plan.get("schema_version")) != ASR_WORKER_EXECUTION_DRY_RUN_SCHEMA_VERSION:
        reasons.append("worker_plan_schema_unexpected")
    if clean(worker_plan.get("job_type")) != ASR_JOB_TYPE:
        reasons.append("worker_plan_job_type_unexpected")
    if clean(worker_plan.get("mode")) != "worker_execution_dry_run":
        reasons.append("worker_plan_mode_unexpected")
    if clean(worker_plan.get("status")) != "worker_envelopes_planned_not_dispatched":
        reasons.append("worker_plan_status_unexpected")
    if not clean(worker_plan.get("approval_ref")):
        reasons.append("worker_plan_approval_ref_missing")
    for key in ("dispatch_allowed", "run_asr", "execution_allowed", "write_outputs"):
        if bool(worker_plan.get(key)):
            reasons.append(f"worker_plan_{key}_must_be_false")
    hard_guards = mapping_or_empty(worker_plan.get("hard_guards"))
    if not hard_guards:
        reasons.append("worker_plan_hard_guards_missing")
    for guard in tuple(DANGEROUS_HARD_GUARDS) + ("dispatch_worker", "write_transcripts"):
        if bool(hard_guards.get(guard)):
            reasons.append(f"worker_plan_hard_guard_{guard}_must_be_false")
    next_stage = mapping_or_empty(worker_plan.get("next_stage_contract"))
    if not bool(next_stage.get("may_create_worker_sandbox_plan")):
        reasons.append("worker_plan_next_stage_sandbox_not_allowed")
    if bool(next_stage.get("may_dispatch_worker_in_this_stage")):
        reasons.append("worker_plan_next_stage_must_not_dispatch_worker")
    if bool(next_stage.get("may_run_asr_in_this_stage")):
        reasons.append("worker_plan_next_stage_must_not_run_asr")
    if not bool(next_stage.get("must_choose_asr_engine_before_execution")):
        reasons.append("worker_plan_engine_choice_requirement_missing")
    workload = mapping_or_empty(worker_plan.get("workload"))
    source_items = optional_int(workload.get("source_items")) or 0
    envelopes = optional_int(workload.get("envelopes")) or 0
    planned = optional_int(workload.get("planned_envelopes")) or 0
    blocked = optional_int(workload.get("blocked_envelopes")) or 0
    if source_items <= 0:
        reasons.append("worker_plan_source_items_empty")
    if envelopes != source_items or planned != source_items:
        reasons.append("worker_plan_envelopes_do_not_match_source_items")
    if blocked:
        reasons.append("worker_plan_has_blocked_envelopes")
    worker_envelopes = worker_plan.get("worker_envelopes")
    if not isinstance(worker_envelopes, SequenceABC) or isinstance(worker_envelopes, (str, bytes)):
        reasons.append("worker_plan_envelopes_missing")
    elif len(worker_envelopes) != envelopes:
        reasons.append("worker_plan_envelope_count_mismatch")
    else:
        reasons.extend(validate_worker_envelopes(worker_envelopes))
    return sorted(set(reasons))


def validate_worker_envelopes(envelopes: Sequence[Any]) -> list[str]:
    reasons: list[str] = []
    for index, envelope in enumerate(envelopes, start=1):
        if not isinstance(envelope, MappingABC):
            reasons.append(f"envelope_{index}_not_object")
            continue
        if clean(envelope.get("action")) != "PLAN_ASR_WORKER_ENVELOPE":
            reasons.append(f"envelope_{index}_action_unexpected")
        command = mapping_or_empty(envelope.get("worker_command_envelope"))
        if command.get("executable") is not None:
            reasons.append(f"envelope_{index}_executable_must_be_null")
        argv = command.get("argv")
        if not isinstance(argv, SequenceABC) or isinstance(argv, (str, bytes)) or len(argv) != 0:
            reasons.append(f"envelope_{index}_argv_must_be_empty")
        for key in ("dry_run_only",):
            if not bool(command.get(key)):
                reasons.append(f"envelope_{index}_{key}_must_be_true")
        for key in ("dispatch_allowed", "run_asr", "write_outputs", "write_runtime_db", "write_crm"):
            if bool(command.get(key)):
                reasons.append(f"envelope_{index}_{key}_must_be_false")
    return reasons


def build_capability_report(
    env: Mapping[str, str],
    module_checker: Callable[[str], bool],
    binary_checker: Callable[[str], Optional[str]],
) -> Mapping[str, Any]:
    env_snapshot = {
        "transcribe_provider": safe_env_text(env, "TRANSCRIBE_PROVIDER") or "mock",
        "dual_transcribe_enabled": safe_env_text(env, "DUAL_TRANSCRIBE_ENABLED") or "false",
        "secondary_transcribe_provider": safe_env_text(env, "SECONDARY_TRANSCRIBE_PROVIDER") or None,
        "openai_api_key_present": bool(clean(env.get("OPENAI_API_KEY"))),
        "openai_transcribe_model": safe_env_text(env, "OPENAI_TRANSCRIBE_MODEL") or "gpt-4o-transcribe",
        "mlx_whisper_model": safe_env_text(env, "MLX_WHISPER_MODEL") or "mlx-community/whisper-large-v3-mlx",
        "gigaam_model": safe_env_text(env, "GIGAAM_MODEL") or "v2_rnnt",
        "gigaam_device": safe_env_text(env, "GIGAAM_DEVICE") or "cpu",
        "gigaam_segment_sec": safe_env_text(env, "GIGAAM_SEGMENT_SEC") or "20",
        "transcribe_language": safe_env_text(env, "TRANSCRIBE_LANGUAGE") or None,
    }
    ffmpeg_path = binary_checker("ffmpeg")
    engines = [
        {
            "engine": "mock",
            "ready": True,
            "counts_as_real_asr": False,
            "network_required": False,
            "module": None,
            "module_available": True,
            "binary_requirements": [],
            "config_refs": {},
            "notes": ["mock provider is useful for tests only and is not a real ASR capability"],
        },
        {
            "engine": "mlx",
            "ready": bool(module_checker("mlx_whisper")),
            "counts_as_real_asr": True,
            "network_required": False,
            "module": "mlx_whisper",
            "module_available": bool(module_checker("mlx_whisper")),
            "binary_requirements": [],
            "config_refs": {
                "model": env_snapshot["mlx_whisper_model"],
                "language": env_snapshot["transcribe_language"],
            },
            "notes": [],
        },
        {
            "engine": "gigaam",
            "ready": bool(module_checker("gigaam")) and bool(ffmpeg_path),
            "counts_as_real_asr": True,
            "network_required": False,
            "module": "gigaam",
            "module_available": bool(module_checker("gigaam")),
            "binary_requirements": [{"name": "ffmpeg", "available": bool(ffmpeg_path), "path": ffmpeg_path}],
            "config_refs": {
                "model": env_snapshot["gigaam_model"],
                "device": env_snapshot["gigaam_device"],
                "segment_sec": env_snapshot["gigaam_segment_sec"],
            },
            "notes": [],
        },
        {
            "engine": "openai",
            "ready": bool(module_checker("openai")) and bool(env_snapshot["openai_api_key_present"]),
            "counts_as_real_asr": True,
            "network_required": True,
            "module": "openai",
            "module_available": bool(module_checker("openai")),
            "binary_requirements": [],
            "config_refs": {
                "model": env_snapshot["openai_transcribe_model"],
                "api_key_present": env_snapshot["openai_api_key_present"],
            },
            "notes": ["API key value is intentionally not included"],
        },
    ]
    for engine in engines:
        missing = []
        if engine["module"] and not engine["module_available"]:
            missing.append(f"missing_python_module:{engine['module']}")
        for binary in engine["binary_requirements"]:
            if not bool(binary.get("available")):
                missing.append(f"missing_binary:{binary.get('name')}")
        if engine["engine"] == "openai" and not env_snapshot["openai_api_key_present"]:
            missing.append("missing_env:OPENAI_API_KEY")
        engine["missing"] = missing
    ready_real = [engine["engine"] for engine in engines if engine["ready"] and engine["counts_as_real_asr"]]
    return {
        "schema_version": ASR_WORKER_SANDBOX_READINESS_SCHEMA_VERSION,
        "mode": "capability_report_read_only",
        "env_refs": env_snapshot,
        "checked_env_names": list(ASR_ENV_REFS),
        "engine_candidates": engines,
        "ready_real_engines": ready_real,
        "asr_engine_ready": bool(ready_real),
        "safety": {
            "imports_asr_modules": False,
            "loads_models": False,
            "runs_asr": False,
            "checks_import_specs_only": True,
            "checks_binary_paths_only": True,
            "secrets_redacted": True,
        },
    }


def build_readiness_actions(
    source_reasons: Sequence[str],
    asr_engine_ready: bool,
    ready_real_engines: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if source_reasons:
        return [
            {
                "action": "BLOCK_ASR_WORKER_SANDBOX_SOURCE_PLAN",
                "reason": "source_worker_plan_failed_readiness_contract",
                "technical_reasons": list(source_reasons),
                "worker_sandbox_ready": False,
                "run_asr": False,
            }
        ]
    if not asr_engine_ready:
        return [
            {
                "action": "BLOCK_ASR_WORKER_SANDBOX_MISSING_ENGINE",
                "reason": "no_real_asr_engine_capability_detected",
                "worker_sandbox_ready": False,
                "run_asr": False,
            }
        ]
    return [
        {
            "action": "PLAN_ASR_WORKER_SANDBOX_READY_DRY_RUN",
            "reason": "source_worker_plan_and_asr_capability_detected_without_execution",
            "ready_engines": [clean(engine.get("engine")) for engine in ready_real_engines],
            "worker_sandbox_ready": True,
            "run_asr": False,
        }
    ]


def build_readiness_payload(
    worker_plan_path: Path,
    worker_plan: Mapping[str, Any],
    worker_plan_sha256: str,
    source_reasons: Sequence[str],
    capability_report: Mapping[str, Any],
    ready_real_engines: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    source_ok = not source_reasons
    engine_ok = bool(ready_real_engines)
    return {
        "schema_version": ASR_WORKER_SANDBOX_READINESS_SCHEMA_VERSION,
        "job_type": ASR_JOB_TYPE,
        "mode": "worker_sandbox_readiness_gate",
        "status": readiness_status(source_ok=source_ok, engine_ok=engine_ok),
        "approval_ref": clean(worker_plan.get("approval_ref")) or None,
        "worker_sandbox_ready": source_ok and engine_ok,
        "asr_engine_ready": engine_ok,
        "dispatch_allowed": False,
        "run_asr": False,
        "write_transcripts": False,
        "write_runtime_db": False,
        "write_crm": False,
        "input_refs": {
            "worker_plan_path": str(worker_plan_path),
            "worker_plan_sha256": worker_plan_sha256,
            "execution_plan_path": mapping_or_empty(worker_plan.get("input_refs")).get("execution_plan_path"),
            "execution_plan_sha256": mapping_or_empty(worker_plan.get("input_refs")).get("execution_plan_sha256"),
            "pack_manifest_path": mapping_or_empty(worker_plan.get("input_refs")).get("pack_manifest_path"),
            "pack_manifest_sha256": mapping_or_empty(worker_plan.get("input_refs")).get("pack_manifest_sha256"),
        },
        "workload": dict(mapping_or_empty(worker_plan.get("workload"))),
        "resource_estimate": dict(mapping_or_empty(worker_plan.get("resource_estimate"))),
        "capability_report": capability_report,
        "actions": list(actions),
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
            "may_create_worker_sandbox_execution_plan": source_ok,
            "may_run_asr_in_this_stage": False,
            "may_dispatch_worker_in_this_stage": False,
            "must_select_engine_before_execution": True,
            "must_require_explicit_execution_approval": True,
            "must_not_write_runtime_db": True,
            "must_not_write_crm": True,
        },
    }


def readiness_status(source_ok: bool, engine_ok: bool) -> str:
    if not source_ok:
        return "blocked_source_worker_plan"
    if not engine_ok:
        return "blocked_missing_asr_engine"
    return "sandbox_ready_dry_run"


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def safe_env_text(env: Mapping[str, str], name: str) -> Optional[str]:
    value = clean(env.get(name))
    if not value:
        return None
    if "KEY" in name or "TOKEN" in name or "SECRET" in name:
        return None
    return value


def resolve_sandbox_readiness_paths(
    product_root: Path,
    worker_plan_path: Path,
    out_dir: Path,
    readiness_report_path: Path,
    out_path: Optional[Path],
) -> Mapping[str, Path]:
    paths = {
        "product_root": product_root.resolve(strict=False),
        "worker_plan_path": worker_plan_path.resolve(strict=False),
        "out_dir": out_dir.resolve(strict=False),
        "readiness_report_path": readiness_report_path.resolve(strict=False),
    }
    if out_path is not None:
        paths["out_path"] = out_path.resolve(strict=False)
    guard_sandbox_readiness_paths(**paths)
    return paths


def guard_sandbox_readiness_paths(
    product_root: Path,
    worker_plan_path: Path,
    out_dir: Path,
    readiness_report_path: Path,
    out_path: Optional[Path] = None,
) -> None:
    for label, path in (
        ("ASR worker plan", worker_plan_path),
        ("ASR worker sandbox readiness output directory", out_dir),
        ("ASR worker sandbox readiness report", readiness_report_path),
    ):
        if "stable_runtime" in path.parts:
            raise ValueError(f"refusing {label} under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not worker_plan_path.exists() or not worker_plan_path.is_file():
        raise FileNotFoundError(f"ASR worker plan not found: {worker_plan_path}")
    if not path_is_relative_to(readiness_report_path, out_dir):
        raise ValueError(f"ASR worker sandbox readiness report must stay under output directory: {out_dir}")
    if out_path is not None:
        if "stable_runtime" in out_path.parts:
            raise ValueError("refusing ASR worker sandbox readiness audit under stable_runtime")
        if not path_is_relative_to(out_path, product_root):
            raise ValueError(f"ASR worker sandbox readiness audit must stay under product root: {product_root}")
        if not path_is_relative_to(out_path, out_dir):
            raise ValueError(f"ASR worker sandbox readiness audit must stay under output directory: {out_dir}")


def sandbox_readiness_safety() -> Mapping[str, bool]:
    return {
        "read_only_inputs": True,
        "reads_worker_plan": True,
        "writes_readiness_report": True,
        "imports_asr_modules": False,
        "loads_models": False,
        "dispatch_worker": False,
        "product_db_writes": False,
        "asset_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "downloads_audio": False,
        "copies_audio": False,
        "hardlinks_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_transcripts": False,
        "write_crm": False,
        "write_tallanto": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def action_counts_for(actions: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return dict(sorted(Counter(clean(action.get("action")) for action in actions).items()))
