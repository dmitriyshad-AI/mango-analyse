from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.repository import ProductRepository
from mango_mvp.productization.test_ingest import path_is_relative_to


SUPERVISOR_DRY_RUN_SCHEMA_VERSION = "supervisor_dry_run_v1"


@dataclass(frozen=True)
class SupervisorStep:
    name: str
    status: str
    would_do: str
    evidence: Mapping[str, Any]
    blocked_by: Sequence[str]

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SupervisorDryRunSummary:
    schema_version: str
    db_path: str
    steps: int
    ready_steps: int
    warning_steps: int
    blocked_steps: int
    runtime_writes_allowed: bool
    asr_allowed: bool
    crm_writes_allowed: bool
    validation_ok: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_supervisor_dry_run_report(
    repo: ProductRepository,
    raw_payload_paths: Optional[Sequence[Path]] = None,
    quarantine_audio_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    repository_summary = repo.summary().to_json_dict()
    raw_payload_stats = [payload_file_stats(path) for path in raw_payload_paths or ()]
    audio_files = count_audio_files(quarantine_audio_dir) if quarantine_audio_dir else 0
    steps = [
        poll_capture_step(raw_payload_stats),
        normalize_dedupe_step(repository_summary),
        archive_provenance_step(repository_summary, raw_payload_stats),
        quarantine_package_step(repository_summary, audio_files),
        manager_identity_step(repository_summary),
    ]
    blocked_steps = sum(1 for step in steps if step.status == "blocked")
    warning_steps = sum(1 for step in steps if step.status == "warning")
    ready_steps = sum(1 for step in steps if step.status == "ready")
    summary = SupervisorDryRunSummary(
        schema_version=SUPERVISOR_DRY_RUN_SCHEMA_VERSION,
        db_path=repository_summary["db_path"],
        steps=len(steps),
        ready_steps=ready_steps,
        warning_steps=warning_steps,
        blocked_steps=blocked_steps,
        runtime_writes_allowed=False,
        asr_allowed=False,
        crm_writes_allowed=False,
        validation_ok=blocked_steps == 0,
    )
    report = {
        "summary": summary.to_json_dict(),
        "repository_summary": repository_summary,
        "raw_payload_stats": raw_payload_stats,
        "quarantine_audio_files": audio_files,
        "steps": [step.to_json_dict() for step in steps],
        "hard_guards": {
            "runtime_db_writes": "blocked",
            "audio_downloads": "blocked_in_this_dry_run",
            "asr": "blocked",
            "ra": "blocked",
            "crm_writes": "blocked",
        },
    }
    if out_path:
        write_json_under_root(report, out_path.resolve(strict=False), repo.out_allowed_root)
    return report


def poll_capture_step(raw_payload_stats: Sequence[Mapping[str, Any]]) -> SupervisorStep:
    rows = sum(int(item.get("rows") or 0) for item in raw_payload_stats)
    status = "ready" if rows else "warning"
    return SupervisorStep(
        name="poll_capture",
        status=status,
        would_do="shadow poll Mango for the configured window and persist only JSON report/payload archive",
        evidence={"raw_payload_files": len(raw_payload_stats), "raw_payload_rows": rows},
        blocked_by=() if rows else ("no_raw_payload_archive_supplied",),
    )


def normalize_dedupe_step(repository_summary: Mapping[str, Any]) -> SupervisorStep:
    rows = int(repository_summary.get("provider_metadata_rows") or 0)
    status = "ready" if rows else "blocked"
    return SupervisorStep(
        name="normalize_dedupe",
        status=status,
        would_do="normalize provider events and dedupe by tenant/provider/event/recording keys",
        evidence={
            "provider_metadata_rows": rows,
            "enriched_view_rows": repository_summary.get("enriched_view_rows"),
        },
        blocked_by=() if rows else ("provider_metadata_missing",),
    )


def archive_provenance_step(
    repository_summary: Mapping[str, Any],
    raw_payload_stats: Sequence[Mapping[str, Any]],
) -> SupervisorStep:
    provider_rows = int(repository_summary.get("provider_metadata_rows") or 0)
    refs = int(repository_summary.get("raw_payload_refs_present") or 0)
    status = "ready" if provider_rows and refs == provider_rows else "warning"
    return SupervisorStep(
        name="archive_provenance",
        status=status,
        would_do="archive raw provider payload refs and verify every captured row has provenance",
        evidence={
            "provider_metadata_rows": provider_rows,
            "raw_payload_refs_present": refs,
            "raw_payload_files": len(raw_payload_stats),
        },
        blocked_by=() if status == "ready" else ("raw_payload_ref_gap",),
    )


def quarantine_package_step(repository_summary: Mapping[str, Any], audio_files: int) -> SupervisorStep:
    provider_rows = int(repository_summary.get("provider_metadata_rows") or 0)
    status = "ready" if provider_rows and audio_files >= provider_rows else "warning"
    return SupervisorStep(
        name="quarantine_package",
        status=status,
        would_do="stage validated audio and metadata into disposable package; no ASR execution",
        evidence={"provider_metadata_rows": provider_rows, "quarantine_audio_files": audio_files},
        blocked_by=() if status == "ready" else ("quarantine_audio_count_below_metadata_count",),
    )


def manager_identity_step(repository_summary: Mapping[str, Any]) -> SupervisorStep:
    manual = int(repository_summary.get("manual_owner_review_items") or 0)
    status = "ready" if manual == 0 else "warning"
    return SupervisorStep(
        name="manager_identity",
        status=status,
        would_do="resolve Mango manager identity and route CRM ownership only after tenant owner config review",
        evidence={
            "manager_extensions": repository_summary.get("manager_extensions"),
            "calls_with_manager_identity": repository_summary.get("calls_with_manager_identity"),
            "calls_with_crm_owner": repository_summary.get("calls_with_crm_owner"),
            "manual_owner_review_items": manual,
        },
        blocked_by=() if manual == 0 else ("tenant_owner_mapping_review_required",),
    )


def payload_file_stats(path: Path) -> Mapping[str, Any]:
    path = path.resolve(strict=False)
    rows = 0
    exists = path.exists() and path.is_file()
    if exists:
        with path.open("r", encoding="utf-8") as fh:
            rows = sum(1 for line in fh if line.strip())
    return {"path": str(path), "exists": exists, "rows": rows}


def count_audio_files(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".mp3", ".wav", ".m4a"})


def write_json_under_root(report: Mapping[str, Any], out_path: Path, out_allowed_root: Path) -> None:
    if not path_is_relative_to(out_path, out_allowed_root):
        raise ValueError(f"supervisor output must stay under allowed root: {out_allowed_root}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
