from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


PROJECT_CLEANUP_MANIFEST_SCHEMA_VERSION = "project_cleanup_manifest_v1"
DEFAULT_PROJECT_CLEANUP_MANIFEST_ROOT = Path("stable_runtime/project_cleanup_manifest_20260511_v1")
DEFAULT_CURRENT_RUNTIME_PATH = Path("stable_runtime/CURRENT_RUNTIME.json")

MANIFEST_COLUMNS = [
    "candidate_path",
    "category",
    "reason",
    "replacement_path",
    "safe_to_quarantine",
    "requires_human_review",
    "entry_type",
    "size_bytes",
    "mtime",
]

RUNTIME_FAMILY_REPLACEMENTS = {
    "sales_master_export_": ("superseded_strict_export", "active_export_root", "superseded by current strict AMO export"),
    "canonical_master_": ("superseded_canonical_master", "canonical_summary", "superseded by current canonical master"),
    "crm_writeback_quality_gate_": (
        "superseded_crm_writeback_quality_gate",
        "crm_quality_summary",
        "superseded by current CRM writeback quality gate",
    ),
    "amo_writeback_queue_": ("superseded_amo_writeback_queue", "amo_queue_summary", "superseded by current AMO queue"),
    "transcript_quality_stage15_export_gate_": (
        "superseded_stage15_gate",
        "stage15_summary",
        "superseded by current Stage15 gate",
    ),
}

QUALITY_RUNTIME_PREFIXES = (
    "transcript_quality_baseline_",
    "transcript_quality_stage14_comparison_",
    "bot_safety_frozen_corpus_",
    "bot_safety_frozen_corpus_validation_",
    "claude_stage15_",
    "non_conversation_claude_fixture_validation_",
)

REVIEW_RUNTIME_PREFIXES = (
    "non_conversation_hard_gate_",
    "external_m1_",
    "final_processing_coverage_report_",
    "asr_gap_report_",
    "asr_ra_monthly_report_",
    "manual_tail_",
    "project_inventory_",
)

ROOT_REVIEW_PREFIXES = (
    "_cleanup_quarantine_",
    "_external_handoffs",
    "_local_archive",
)

LOCAL_JUNK_NAMES = {".DS_Store", "Thumbs.db"}
DATE_TOKEN_RE = re.compile(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)")


@dataclass(frozen=True)
class ProjectCleanupManifestConfig:
    project_root: Path
    out_root: Path = DEFAULT_PROJECT_CLEANUP_MANIFEST_ROOT
    current_runtime_path: Optional[Path] = DEFAULT_CURRENT_RUNTIME_PATH
    generated_at: Optional[datetime] = None
    fresh_audit_days: int = 1


@dataclass(frozen=True)
class CleanupCandidate:
    candidate_path: str
    category: str
    reason: str
    replacement_path: str
    safe_to_quarantine: bool
    requires_human_review: bool
    entry_type: str
    size_bytes: int
    mtime: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_csv_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["safe_to_quarantine"] = _bool_text(self.safe_to_quarantine)
        row["requires_human_review"] = _bool_text(self.requires_human_review)
        return row


def build_project_cleanup_manifest(config: ProjectCleanupManifestConfig) -> dict[str, Any]:
    """Build a conservative cleanup candidate manifest without moving or deleting files."""

    project_root = config.project_root.expanduser().resolve(strict=False)
    out_root = _resolve_under_project(project_root, config.out_root)
    generated_at = config.generated_at or datetime.now(timezone.utc)
    current_runtime_path = _resolve_optional(project_root, config.current_runtime_path)
    current_runtime = _load_json_if_exists(current_runtime_path)
    runtime_paths = _runtime_paths(current_runtime)
    protected_paths = _protected_paths(project_root, runtime_paths, out_root)
    replacement_by_key = _replacement_paths(project_root, runtime_paths)
    fresh_cutoff = generated_at.date() - timedelta(days=max(0, config.fresh_audit_days))

    candidates: list[CleanupCandidate] = []
    candidates.extend(
        _scan_project_root(
            project_root=project_root,
            protected_paths=protected_paths,
            replacement_by_key=replacement_by_key,
        )
    )
    candidates.extend(
        _scan_stable_runtime(
            project_root=project_root,
            protected_paths=protected_paths,
            replacement_by_key=replacement_by_key,
        )
    )
    candidates.extend(
        _scan_audits(
            project_root=project_root,
            protected_paths=protected_paths,
            generated_date=generated_at.date(),
            fresh_cutoff=fresh_cutoff,
        )
    )
    candidates = _dedupe_candidates(candidates)

    out_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "manifest_csv": out_root / "manifest.csv",
        "manifest_json": out_root / "manifest.json",
        "summary_json": out_root / "summary.json",
    }
    _write_manifest_csv(outputs["manifest_csv"], candidates)
    _write_json(outputs["manifest_json"], [item.to_json_dict() for item in candidates])

    category_counts = dict(Counter(item.category for item in candidates).most_common())
    summary = {
        "schema_version": PROJECT_CLEANUP_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "out_root": str(out_root),
        "current_runtime_path": str(current_runtime_path) if current_runtime_path else "",
        "current_runtime_loaded": bool(current_runtime),
        "fresh_audit_cutoff": fresh_cutoff.isoformat(),
        "candidate_rows": len(candidates),
        "safe_to_quarantine_rows": sum(1 for item in candidates if item.safe_to_quarantine),
        "requires_human_review_rows": sum(1 for item in candidates if item.requires_human_review),
        "category_counts": category_counts,
        "protected_runtime_paths": sorted(_rel(path, project_root) for path in protected_paths if _is_under(path, project_root)),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "safety": {
            "read_only_scan": True,
            "deletes_files": False,
            "moves_files": False,
            "quarantines_files": False,
            "writes_only_report_artifacts": True,
            "destructive_operations_available": False,
        },
    }
    _write_json(outputs["summary_json"], summary)
    return summary


def _scan_project_root(
    *,
    project_root: Path,
    protected_paths: set[Path],
    replacement_by_key: Mapping[str, Path],
) -> list[CleanupCandidate]:
    rows: list[CleanupCandidate] = []
    for path in sorted(project_root.iterdir(), key=lambda item: item.name):
        if path.name in {".git", "stable_runtime", "audits"}:
            continue
        if _is_protected(path, protected_paths):
            continue
        candidate = _classify_root_entry(path, project_root, replacement_by_key)
        if candidate:
            rows.append(candidate)
    return rows


def _scan_stable_runtime(
    *,
    project_root: Path,
    protected_paths: set[Path],
    replacement_by_key: Mapping[str, Path],
) -> list[CleanupCandidate]:
    stable_runtime = project_root / "stable_runtime"
    if not stable_runtime.exists():
        return []
    rows: list[CleanupCandidate] = []
    for path in sorted(stable_runtime.iterdir(), key=lambda item: item.name):
        if _is_protected(path, protected_paths):
            continue
        if path.name in {"README.md", "SNAPSHOT_CREATED_AT.txt"}:
            continue
        if path.name == "amocrm_runtime":
            continue
        candidate = _classify_stable_runtime_entry(path, project_root, replacement_by_key)
        if candidate:
            rows.append(candidate)
    return rows


def _scan_audits(
    *,
    project_root: Path,
    protected_paths: set[Path],
    generated_date: date,
    fresh_cutoff: date,
) -> list[CleanupCandidate]:
    audits_root = project_root / "audits"
    if not audits_root.exists():
        return []
    rows: list[CleanupCandidate] = []
    for parent_name in ("_inbox", "_results"):
        parent = audits_root / parent_name
        if not parent.exists():
            continue
        for path in sorted(parent.iterdir(), key=lambda item: item.name):
            if path.name == ".gitkeep" or _is_protected(path, protected_paths):
                continue
            if _is_fresh_audit_pack(path.name, generated_date=generated_date, fresh_cutoff=fresh_cutoff):
                continue
            rows.append(
                _candidate(
                    path=path,
                    project_root=project_root,
                    category="historical_audit_pack",
                    reason=f"audit pack older than fresh cutoff {fresh_cutoff.isoformat()}",
                    replacement_path="audits/_inbox or audits/_results current active packs",
                    safe_to_quarantine=True,
                    requires_human_review=True,
                )
            )
    return rows


def _classify_root_entry(
    path: Path,
    project_root: Path,
    replacement_by_key: Mapping[str, Path],
) -> CleanupCandidate | None:
    if path.name in LOCAL_JUNK_NAMES:
        return _candidate(
            path=path,
            project_root=project_root,
            category="local_os_metadata",
            reason="local OS metadata file, not a project artifact",
            replacement_path="",
            safe_to_quarantine=True,
            requires_human_review=False,
        )
    if path.suffix.lower() in {".log", ".tmp", ".bak"}:
        return _candidate(
            path=path,
            project_root=project_root,
            category="root_scratch_file",
            reason="top-level scratch/log/backup file",
            replacement_path="",
            safe_to_quarantine=True,
            requires_human_review=True,
        )
    if any(path.name.startswith(prefix) for prefix in ROOT_REVIEW_PREFIXES):
        return _candidate(
            path=path,
            project_root=project_root,
            category="root_archive_or_handoff_review",
            reason="top-level archive/handoff/quarantine-like artifact; review before moving",
            replacement_path=_rel_path(replacement_by_key.get("quarantine_manifest"), project_root),
            safe_to_quarantine=False,
            requires_human_review=True,
        )
    if path.suffix.lower() in {".docx", ".pptx", ".xlsx"}:
        return _candidate(
            path=path,
            project_root=project_root,
            category="root_binary_document_review",
            reason="top-level binary document artifact should be reviewed before cleanup",
            replacement_path="docs/",
            safe_to_quarantine=False,
            requires_human_review=True,
        )
    return None


def _classify_stable_runtime_entry(
    path: Path,
    project_root: Path,
    replacement_by_key: Mapping[str, Path],
) -> CleanupCandidate | None:
    if path.name in LOCAL_JUNK_NAMES:
        return _candidate(
            path=path,
            project_root=project_root,
            category="local_os_metadata",
            reason="local OS metadata file, not a runtime artifact",
            replacement_path="",
            safe_to_quarantine=True,
            requires_human_review=False,
        )
    if path.suffix.lower() in {".log", ".tmp", ".bak"} or path.name.endswith(".prepare.log"):
        return _candidate(
            path=path,
            project_root=project_root,
            category="runtime_log_or_scratch",
            reason="runtime log/scratch artifact",
            replacement_path="",
            safe_to_quarantine=True,
            requires_human_review=True,
        )
    for prefix, (category, replacement_key, reason) in RUNTIME_FAMILY_REPLACEMENTS.items():
        if path.name.startswith(prefix):
            return _candidate(
                path=path,
                project_root=project_root,
                category=category,
                reason=reason,
                replacement_path=_rel_path(_replacement_parent(replacement_by_key.get(replacement_key)), project_root),
                safe_to_quarantine=True,
                requires_human_review=True,
            )
    if path.name.startswith(QUALITY_RUNTIME_PREFIXES):
        return _candidate(
            path=path,
            project_root=project_root,
            category="superseded_quality_artifact",
            reason="quality/audit runtime output not pinned by CURRENT_RUNTIME.json",
            replacement_path=_rel_path(_replacement_parent(replacement_by_key.get("stage15_summary")), project_root),
            safe_to_quarantine=True,
            requires_human_review=True,
        )
    if path.name.startswith(REVIEW_RUNTIME_PREFIXES):
        return _candidate(
            path=path,
            project_root=project_root,
            category="runtime_manual_review_required",
            reason="runtime artifact may contain evidence or recovery material; review before quarantine",
            replacement_path=_rel_path(_replacement_parent(replacement_by_key.get("active_export_root")), project_root),
            safe_to_quarantine=False,
            requires_human_review=True,
        )
    if path.name.startswith("operator_status_") or path.name.startswith("call_processing_readiness_"):
        return _candidate(
            path=path,
            project_root=project_root,
            category="superseded_status_report",
            reason="status/readiness report can be regenerated from current runtime contract",
            replacement_path=_rel_path(replacement_by_key.get("runtime_source"), project_root),
            safe_to_quarantine=True,
            requires_human_review=True,
        )
    return None


def _candidate(
    *,
    path: Path,
    project_root: Path,
    category: str,
    reason: str,
    replacement_path: str,
    safe_to_quarantine: bool,
    requires_human_review: bool,
) -> CleanupCandidate:
    stat = path.stat()
    return CleanupCandidate(
        candidate_path=_rel(path, project_root),
        category=category,
        reason=reason,
        replacement_path=replacement_path,
        safe_to_quarantine=safe_to_quarantine,
        requires_human_review=requires_human_review,
        entry_type="dir" if path.is_dir() else "file",
        size_bytes=_path_size_bytes(path),
        mtime=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    )


def _runtime_paths(current_runtime: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, value in _mapping(current_runtime.get("paths")).items():
        path = _path_from_value(value)
        if path:
            paths[key] = path
    return paths


def _replacement_paths(project_root: Path, runtime_paths: Mapping[str, Path]) -> dict[str, Path]:
    replacements = {key: value for key, value in runtime_paths.items()}
    replacements["runtime_source"] = project_root / "stable_runtime" / "CURRENT_RUNTIME.json"
    return replacements


def _protected_paths(project_root: Path, runtime_paths: Mapping[str, Path], out_root: Path) -> set[Path]:
    protected = {
        project_root / "stable_runtime" / "CURRENT_RUNTIME.json",
        project_root / "stable_runtime" / "CANONICAL_EXPORT.txt",
        project_root / "stable_runtime" / "amocrm_runtime",
        out_root,
    }
    for path in runtime_paths.values():
        protected.add(path)
        runtime_child = _surface_child(project_root / "stable_runtime", path)
        if runtime_child:
            protected.add(runtime_child)
        audit_child = _surface_child(project_root / "audits", path)
        if audit_child:
            protected.add(audit_child)
        root_child = _surface_child(project_root, path)
        if root_child:
            if root_child.name not in {"stable_runtime", "audits"}:
                protected.add(root_child)
    return {item.expanduser().resolve(strict=False) for item in protected}


def _surface_child(surface_root: Path, path: Path) -> Path | None:
    try:
        rel = path.expanduser().resolve(strict=False).relative_to(surface_root.expanduser().resolve(strict=False))
    except ValueError:
        return None
    if not rel.parts:
        return surface_root
    return surface_root / rel.parts[0]


def _is_protected(path: Path, protected_paths: set[Path]) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    for protected in protected_paths:
        if resolved == protected or _is_under(protected, resolved):
            return True
    return False


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.expanduser().resolve(strict=False).relative_to(parent.expanduser().resolve(strict=False))
        return True
    except ValueError:
        return False


def _is_fresh_audit_pack(name: str, *, generated_date: date, fresh_cutoff: date) -> bool:
    dates = _dates_from_name(name)
    if not dates:
        return False
    return max(dates) >= fresh_cutoff and max(dates) <= generated_date


def _dates_from_name(name: str) -> list[date]:
    dates: list[date] = []
    for year, month, day in DATE_TOKEN_RE.findall(name):
        try:
            dates.append(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    return dates


def _dedupe_candidates(candidates: Iterable[CleanupCandidate]) -> list[CleanupCandidate]:
    by_path: dict[str, CleanupCandidate] = {}
    for item in candidates:
        by_path[item.candidate_path] = item
    return sorted(by_path.values(), key=lambda item: (item.category, item.candidate_path))


def _write_manifest_csv(path: Path, candidates: list[CleanupCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for item in candidates:
            writer.writerow(item.to_csv_dict())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json_if_exists(path: Path | None) -> Mapping[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _path_from_value(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve(strict=False)


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        try:
            total += child.stat().st_size
        except OSError:
            continue
    return total


def _replacement_parent(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_dir() else path.parent


def _resolve_under_project(project_root: Path, path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def _resolve_optional(project_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return _resolve_under_project(project_root, path)


def _rel_path(path: Path | None, project_root: Path) -> str:
    if path is None:
        return ""
    return _rel(path, project_root)


def _rel(path: Path, project_root: Path) -> str:
    try:
        return str(path.expanduser().resolve(strict=False).relative_to(project_root.expanduser().resolve(strict=False)))
    except ValueError:
        return str(path.expanduser().resolve(strict=False))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "CleanupCandidate",
    "DEFAULT_PROJECT_CLEANUP_MANIFEST_ROOT",
    "MANIFEST_COLUMNS",
    "PROJECT_CLEANUP_MANIFEST_SCHEMA_VERSION",
    "ProjectCleanupManifestConfig",
    "build_project_cleanup_manifest",
]
