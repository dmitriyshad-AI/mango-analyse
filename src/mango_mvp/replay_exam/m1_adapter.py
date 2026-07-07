from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from .provider_adapter import SCRUBBED_ROOT, assert_scrubbed_cases_path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_status_clean(root: Path) -> bool:
    status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True)
    return not status.strip()


def _command_arg(command: Sequence[str], flag: str) -> str:
    try:
        index = list(command).index(flag)
    except ValueError as exc:
        raise ValueError(f"M1 replay command must include {flag}") from exc
    if index + 1 >= len(command):
        raise ValueError(f"M1 replay command flag {flag} requires a value")
    value = str(command[index + 1]).strip()
    if not value:
        raise ValueError(f"M1 replay command flag {flag} requires a non-empty value")
    return value


def validate_replay_m1_command(command: Sequence[str]) -> Path:
    """Package lint: real M1 replay must run from local scrubbed cases, not package files."""
    set_arg = _command_arg(command, "--set")
    return assert_scrubbed_cases_path(Path(set_arg))


def _case_stats(set_path: Path) -> dict[str, object]:
    dialogs: set[str] = set()
    segments: dict[str, int] = {}
    brands: dict[str, int] = {}
    turns = 0
    for line in set_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        turns += 1
        dialogs.add(str(item.get("dialog_id") or item.get("dialog_key_masked") or ""))
        segment = str(item.get("segment") or "unknown")
        brand = str(item.get("brand") or "unknown")
        segments[segment] = segments.get(segment, 0) + 1
        brands[brand] = brands.get(brand, 0) + 1
    return {
        "turns": turns,
        "dialogs": len({dialog for dialog in dialogs if dialog}),
        "segments": dict(sorted(segments.items())),
        "brands": dict(sorted(brands.items())),
    }


def build_replay_m1_manifest(
    *,
    set_path: Path,
    out_path: Path,
    command: Sequence[str],
    repo_root: Path | None = None,
    live_head: str = "",
    snapshot_path: Path | None = None,
    bundle_path: Path | None = None,
    source_head_path: Path | None = None,
    pii_report_path: Path | None = None,
    raw_manifest_path: Path | None = None,
    retention_manifest_path: Path | None = None,
    budgets: Mapping[str, int] | None = None,
) -> Path:
    root = (repo_root or Path.cwd()).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    set_path = set_path.resolve()
    out_path = out_path.resolve()
    snapshot_resolved = snapshot_path.resolve() if snapshot_path is not None else None
    bundle_resolved = bundle_path.resolve() if bundle_path is not None else None
    source_head_resolved = source_head_path.resolve() if source_head_path is not None else None
    pii_resolved = pii_report_path.resolve() if pii_report_path is not None else None
    raw_manifest_resolved = raw_manifest_path.resolve() if raw_manifest_path is not None else None
    retention_resolved = retention_manifest_path.resolve() if retention_manifest_path is not None else None
    command_set_path = validate_replay_m1_command(command)
    pii_leak_count = None
    if pii_resolved is not None and pii_resolved.exists():
        pii_payload = json.loads(pii_resolved.read_text(encoding="utf-8"))
        pii_leak_count = int(pii_payload.get("leak_count") or 0)
    payload = {
        "schema_version": "wappi_replay_m1_manifest_v2",
        "metric": "chat_only_replay",
        "source_head": head,
        "eval_head": head,
        "live_head": live_head,
        "git_status_clean": _git_status_clean(root),
        "set_path": str(set_path),
        "m1_scrubbed_set_path": str(command_set_path),
        "set_sha256": _sha256(set_path),
        "case_stats": _case_stats(set_path),
        "command": list(command),
        "budgets": dict(budgets or {}),
        "live_writes_allowed": False,
        "raw_included": False,
        "scrubbed_only": True,
        "m1_prompt_requirements": {
            "copy_package_set_to_scrubbed_path_before_run": True,
            "verify_copied_set_sha256": True,
            "scrubbed_set_root": str(SCRUBBED_ROOT),
            "include_scrubbed_set_copy_in_retention": True,
        },
        "parallelism": "dialogs_only",
        "real_provider_parallel_cap": 2,
        "m1_contract": {
            "dry_check_first": True,
            "one_full_run": True,
            "no_retries": True,
            "return_validity_and_paths_without_verdict": True,
            "judge_second_pass": True,
        },
    }
    if snapshot_resolved is not None:
        payload["snapshot_path"] = str(snapshot_resolved)
        payload["snapshot_sha256"] = _sha256(snapshot_resolved)
    if bundle_resolved is not None:
        payload["bundle_path"] = str(bundle_resolved)
        payload["bundle_sha256"] = _sha256(bundle_resolved)
    if source_head_resolved is not None:
        payload["source_head_path"] = str(source_head_resolved)
        payload["source_head_sha256"] = _sha256(source_head_resolved)
    if pii_resolved is not None:
        payload["pii_report_path"] = str(pii_resolved)
        payload["pii_leak_count"] = pii_leak_count
    if raw_manifest_resolved is not None:
        payload["raw_manifest_path_local_only"] = str(raw_manifest_resolved)
    if retention_resolved is not None:
        payload["retention_manifest_path"] = str(retention_resolved)
        retention_paths = [str(set_path.parent), str(command_set_path.parent)]
        if retention_resolved.exists():
            retention_payload = json.loads(retention_resolved.read_text(encoding="utf-8"))
            raw_paths = retention_payload.get("suggested_delete_paths_after_exam")
            if isinstance(raw_paths, list):
                retention_paths = [str(item) for item in raw_paths if str(item).strip()]
                if str(command_set_path.parent) not in retention_paths:
                    retention_paths.append(str(command_set_path.parent))
        payload["retention_delete_command"] = "rm -rf -- " + " ".join(retention_paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
