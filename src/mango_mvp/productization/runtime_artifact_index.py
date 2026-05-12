from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


RUNTIME_ARTIFACT_INDEX_SCHEMA_VERSION = "runtime_artifact_index_v1"
DEFAULT_RUNTIME_ARTIFACT_INDEX_ROOT = Path("stable_runtime/runtime_artifact_index_20260511_v1")


@dataclass(frozen=True)
class RuntimeArtifactIndexSummary:
    schema_version: str
    generated_at: str
    project_root: str
    entries: int
    active_current: int
    blocked: int
    audit_only: int
    legacy_candidates: int
    unknown_review: int
    invalid_json_artifacts: int
    read_only_scan: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_runtime_artifact_index(
    *,
    project_root: Path,
    out_root: Optional[Path] = None,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    """Build a read-only index of stable_runtime artifacts.

    The index is intentionally conservative: it never deletes or moves files,
    and it marks evidence from dry-run/live/readback as protected audit-only.
    """

    project_root = project_root.resolve(strict=False)
    stable = project_root / "stable_runtime"
    out_root = _resolve_optional(project_root, out_root or DEFAULT_RUNTIME_ARTIFACT_INDEX_ROOT)
    now = generated_at or datetime.now(timezone.utc)
    current_runtime_path = stable / "CURRENT_RUNTIME.json"
    current_runtime = _load_json_if_exists(current_runtime_path)
    current_paths = _current_runtime_paths(current_runtime)
    cleanup_manifest_paths = _cleanup_manifest_paths(project_root)
    latest_operator = _latest_artifact(stable, "operator_status_*/operator_status.json")
    latest_waiting = _latest_artifact(stable, "amo_waiting_autonomous_work_*/summary.json")

    entries: list[Mapping[str, Any]] = []
    if stable.exists():
        artifact_paths = list(stable.iterdir())
        contact_writebacks = stable / "amocrm_runtime" / "contact_writebacks"
        if contact_writebacks.exists():
            artifact_paths.extend(path for path in contact_writebacks.iterdir() if path.is_dir())
        for path in sorted(set(artifact_paths), key=lambda item: str(item)):
            entries.append(
                _classify_artifact(
                    project_root=project_root,
                    path=path,
                    current_paths=current_paths,
                    cleanup_manifest_paths=cleanup_manifest_paths,
                    latest_operator=latest_operator,
                    latest_waiting=latest_waiting,
                )
            )

    counters = _counters(entries)
    summary = RuntimeArtifactIndexSummary(
        schema_version=RUNTIME_ARTIFACT_INDEX_SCHEMA_VERSION,
        generated_at=now.isoformat(timespec="seconds"),
        project_root=str(project_root),
        entries=len(entries),
        active_current=counters["active_current"],
        blocked=counters["blocked"],
        audit_only=counters["audit_only"],
        legacy_candidates=counters["legacy_candidate"],
        unknown_review=counters["unknown_review"],
        invalid_json_artifacts=sum(1 for entry in entries if entry.get("valid_json") is False),
        read_only_scan=True,
    )
    payload = {
        "summary": summary.to_json_dict(),
        "source_of_truth": {
            "current_runtime": str(current_runtime_path),
            "latest_operator_status": str(latest_operator) if latest_operator else None,
            "latest_waiting_autonomous_work": str(latest_waiting) if latest_waiting else None,
            "cleanup_manifest_paths": sorted(str(path) for path in cleanup_manifest_paths),
        },
        "entries": entries,
        "safety": {
            "read_only_scan": True,
            "moves_files": False,
            "deletes_files": False,
            "writes_crm": False,
            "writes_runtime_db": False,
        },
    }
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)
        _write_json(out_root / "summary.json", payload["summary"])
        _write_json(out_root / "artifact_index.json", payload)
        _write_entries_csv(out_root / "artifact_index.csv", entries)
        (out_root / "README.md").write_text(render_runtime_artifact_index_markdown(payload), encoding="utf-8")
    return payload


def render_runtime_artifact_index_markdown(payload: Mapping[str, Any]) -> str:
    summary = _mapping(payload.get("summary"))
    lines = [
        "# Runtime Artifact Index",
        "",
        f"Generated at: `{summary.get('generated_at')}`",
        "",
        "## Summary",
        "",
        f"- Entries: `{summary.get('entries')}`",
        f"- Active/current: `{summary.get('active_current')}`",
        f"- Blocked: `{summary.get('blocked')}`",
        f"- Audit-only: `{summary.get('audit_only')}`",
        f"- Legacy candidates: `{summary.get('legacy_candidates')}`",
        f"- Unknown/review: `{summary.get('unknown_review')}`",
        f"- Invalid JSON artifacts: `{summary.get('invalid_json_artifacts')}`",
        "",
        "## Policy",
        "",
        "- This is a read-only index.",
        "- It does not delete, move, or quarantine files.",
        "- Audit evidence is protected and must not be used as a direct live-write input.",
        "- Legacy candidates still require the cleanup manifest workflow before quarantine.",
    ]
    return "\n".join(lines) + "\n"


def _classify_artifact(
    *,
    project_root: Path,
    path: Path,
    current_paths: set[Path],
    cleanup_manifest_paths: set[Path],
    latest_operator: Optional[Path],
    latest_waiting: Optional[Path],
) -> Mapping[str, Any]:
    summary_path = _summary_path_for_artifact(path)
    summary = _load_json_if_exists(summary_path) if summary_path.exists() else {}
    valid_json = None
    if summary_path.exists() and summary_path.suffix == ".json":
        valid_json = bool(summary)
    rel = _relative(project_root, path)
    role = _role(path)
    is_current = path in current_paths or any(_path_is_relative_to(current_path, path) for current_path in current_paths)
    if latest_operator and (path == latest_operator.parent or path == latest_operator):
        is_current = True
        role = "operator_status"
    if latest_waiting and (path == latest_waiting.parent or path == latest_waiting):
        is_current = True
        role = "waiting_autonomous_work"
    is_cleanup_candidate = path in cleanup_manifest_paths or any(_path_is_relative_to(candidate, path) for candidate in cleanup_manifest_paths)
    blocked_reasons = _blocked_reasons(path, summary, valid_json)
    if is_current:
        category = "active_current"
    elif blocked_reasons:
        category = "blocked"
    elif "amocrm_runtime" in path.parts or "claude" in path.name.lower() or "audit" in path.name.lower():
        category = "audit_only"
    elif is_cleanup_candidate or "202604" in path.name or "legacy" in path.name.lower():
        category = "legacy_candidate"
    else:
        category = "unknown_review"
    can_feed_live = category == "active_current" and not blocked_reasons and role in {
        "strict_amo_export",
        "amo_writeback_queue",
    }
    return {
        "path": rel,
        "category": category,
        "role": role,
        "protected": category in {"active_current", "audit_only", "blocked"},
        "can_feed_live": can_feed_live,
        "blocked_reasons": blocked_reasons,
        "valid_json": valid_json,
        "summary_path": _relative(project_root, summary_path) if summary_path.exists() else "",
        "superseded_by": "",
    }


def _role(path: Path) -> str:
    name = path.name
    if "contact_writebacks" in path.parts:
        return "amocrm_contact_writeback_run"
    if name == "CURRENT_RUNTIME.json":
        return "current_runtime_contract"
    if name == "CANONICAL_EXPORT.txt":
        return "canonical_export_pointer"
    if name.startswith("sales_master_export_"):
        return "strict_amo_export" if "crm_text_quality_strict" in name else "amo_export"
    if name.startswith("canonical_master_"):
        return "canonical_master"
    if name.startswith("transcript_quality_stage15_export_gate_"):
        return "stage15_transcript_gate"
    if name.startswith("crm_writeback_quality_gate_"):
        return "crm_writeback_quality_gate"
    if name.startswith("amo_writeback_queue_"):
        return "amo_writeback_queue"
    if name.startswith("operator_status_"):
        return "operator_status"
    if name.startswith("amo_waiting_autonomous_work_"):
        return "waiting_autonomous_work"
    if name.startswith("amo_duplicate_"):
        return "duplicate_resolution"
    if name.startswith("amocrm_runtime"):
        return "amocrm_runtime_evidence"
    if name.startswith("project_cleanup_manifest_"):
        return "cleanup_manifest"
    return "runtime_artifact"


def _summary_path_for_artifact(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("summary.json", "contact_writeback_summary.json", "readback_summary.json"):
        candidate = path / name
        if candidate.exists():
            return candidate
    return path / "summary.json"


def _blocked_reasons(path: Path, summary: Mapping[str, Any], valid_json: Optional[bool]) -> list[str]:
    reasons: list[str] = []
    if valid_json is False:
        reasons.append("invalid_json")
    if summary.get("passed") is False:
        reasons.append("passed_false")
    if summary.get("preflight_failed") is True:
        reasons.append("preflight_failed")
    if _int(summary.get("blocked_rows")) > 0:
        reasons.append("blocked_rows_present")
    if summary.get("stage50_preflight_allowed") is False or summary.get("stage86_preflight_allowed") is False:
        reasons.append("stage_rollout_blocked")
    if summary.get("status") in {"pending_not_run", "waiting_for_staff_done_and_recheck"}:
        reasons.append(str(summary.get("status")))
    if "contact_writebacks" in path.parts and _int(summary.get("failed")) > 0:
        reasons.append("writeback_failed_rows")
    return reasons


def _current_runtime_paths(current_runtime: Mapping[str, Any]) -> set[Path]:
    paths = set()
    for value in _mapping(current_runtime.get("paths")).values():
        if isinstance(value, str) and value.strip():
            paths.add(Path(value).expanduser().resolve(strict=False))
    return paths


def _cleanup_manifest_paths(project_root: Path) -> set[Path]:
    candidates = sorted((project_root / "stable_runtime").glob("project_cleanup_manifest_*/manifest.csv"))
    if not candidates:
        return set()
    paths: set[Path] = set()
    with candidates[-1].open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row.get("path") or row.get("relative_path") or row.get("original_path") or ""
            if raw.strip():
                paths.add((project_root / raw).resolve(strict=False))
    return paths


def _counters(entries: list[Mapping[str, Any]]) -> Mapping[str, int]:
    counters = {
        "active_current": 0,
        "blocked": 0,
        "audit_only": 0,
        "legacy_candidate": 0,
        "unknown_review": 0,
    }
    for entry in entries:
        category = str(entry.get("category") or "unknown_review")
        counters[category] = counters.get(category, 0) + 1
    return counters


def _latest_artifact(root: Path, pattern: str) -> Optional[Path]:
    candidates = [path for path in root.glob(pattern) if path.exists()]
    return max(candidates, key=lambda item: (item.stat().st_mtime_ns, str(item))) if candidates else None


def _load_json_if_exists(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_entries_csv(path: Path, entries: list[Mapping[str, Any]]) -> None:
    fieldnames = ["path", "category", "role", "protected", "can_feed_live", "blocked_reasons", "valid_json", "summary_path", "superseded_by"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = dict(entry)
            row["blocked_reasons"] = " | ".join(entry.get("blocked_reasons") or [])
            writer.writerow(row)


def _resolve_optional(project_root: Path, path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def _relative(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(project_root))
    except ValueError:
        return str(path.resolve(strict=False))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


__all__ = [
    "RUNTIME_ARTIFACT_INDEX_SCHEMA_VERSION",
    "DEFAULT_RUNTIME_ARTIFACT_INDEX_ROOT",
    "build_runtime_artifact_index",
    "render_runtime_artifact_index_markdown",
]
