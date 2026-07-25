#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.publish_snapshot.common import (
    PublishConfig,
    add_common_args,
    disk_report,
    finish_cli,
    foreign_key_check,
    git_head,
    git_status_short,
    classify_publish_worktree_status,
    load_config,
    quick_check,
    report_base,
    backup_plan_report,
    schema_diff,
    sha256_file,
    table_counts,
    wal_sidecar_report,
)


# Fail-closed if a nightly manifest's published_at is ahead of "now" by more
# than this many minutes. Small clock drift between hosts is expected; a
# manifest minutes/hours in the future is a clock fault or a forged/tampered
# manifest, not freshness.
MANIFEST_FUTURE_TOLERANCE_MINUTES = 5.0


def nightly_manifest_report(cfg: PublishConfig) -> dict[str, Any]:
    """Prove that the staging DB being published reflects a nightly run that
    fully succeeded, not a stale or partially-failed night.

    Reads the nightly service's own ``latest_customer_timeline_snapshot.json``
    (only ever (re)written by nightly_service when ``latest_published`` was
    True for that run -- see NightlyServiceConfig.run_nightly_service), then
    follows its embedded ``service_report_path`` back to that run's
    ``service_report.json`` to check ``failed_required_steps``/
    ``partial_failure``/``overall_status`` directly from source, and
    cross-checks the manifest's recorded sha256/size/counts against the
    staging DB actually sitting on disk right now. Every failure mode is
    collected in ``reasons`` and folds into ``ok`` -- fail-closed: anything
    unreadable, mismatched or missing counts as not-ok.
    """
    manifest_path = cfg.nightly_manifest_path
    reasons: list[str] = []
    report: dict[str, Any] = {
        "path": str(manifest_path),
        "max_age_hours": cfg.max_manifest_age_hours,
    }
    if not manifest_path.is_file():
        reasons.append("nightly_manifest_missing")
        report.update({"exists": False, "ok": False, "reasons": reasons})
        return report
    report["exists"] = True
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        reasons.append("nightly_manifest_unreadable")
        report.update({"ok": False, "reasons": reasons})
        return report
    if not isinstance(manifest, Mapping):
        reasons.append("nightly_manifest_invalid")
        report.update({"ok": False, "reasons": reasons})
        return report

    report["run_id"] = manifest.get("run_id")
    report["manifest_timeline_db"] = manifest.get("timeline_db")
    report["published_at"] = manifest.get("published_at")

    age_hours = _manifest_age_hours(manifest.get("published_at"))
    report["manifest_age_hours"] = round(age_hours, 3) if age_hours is not None else None
    report["manifest_future_tolerance_minutes"] = MANIFEST_FUTURE_TOLERANCE_MINUTES
    future_tolerance_hours = MANIFEST_FUTURE_TOLERANCE_MINUTES / 60.0
    if age_hours is None:
        reasons.append("nightly_manifest_published_at_invalid")
        report["fresh"] = False
        report["future_dated"] = False
    elif age_hours < -future_tolerance_hours:
        # published_at is ahead of "now" beyond clock-drift tolerance -- a
        # naive max(0, age) clamp used to silently read this as age=0 (i.e.
        # "fresh"). Must fail closed instead: clock skew or a forged manifest.
        report["fresh"] = False
        report["future_dated"] = True
        reasons.append("nightly_manifest_published_at_future")
    else:
        fresh = max(0.0, age_hours) <= cfg.max_manifest_age_hours
        report["fresh"] = fresh
        report["future_dated"] = False
        if not fresh:
            reasons.append("nightly_manifest_stale")

    manifest_db_matches = report["manifest_timeline_db"] == str(cfg.staging_db)
    report["manifest_timeline_db_matches_staging"] = manifest_db_matches
    if not manifest_db_matches:
        reasons.append("nightly_manifest_timeline_db_mismatch")

    files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
    sqlite_meta = files.get("sqlite") if isinstance(files.get("sqlite"), Mapping) else {}
    manifest_sha256 = sqlite_meta.get("sha256")
    manifest_size = sqlite_meta.get("size_bytes")
    staging_exists = cfg.staging_db.exists()
    staging_sha256 = sha256_file(cfg.staging_db) if staging_exists else None
    staging_size = cfg.staging_db.stat().st_size if staging_exists else None
    report.update(
        {
            "manifest_sha256": manifest_sha256,
            "staging_sha256": staging_sha256,
            "manifest_size_bytes": manifest_size,
            "staging_size_bytes": staging_size,
        }
    )
    sha_match = bool(staging_exists and manifest_sha256 and manifest_sha256 == staging_sha256)
    report["sha256_match"] = sha_match
    if not sha_match:
        reasons.append("nightly_manifest_sha256_mismatch")
    size_match = bool(staging_exists and manifest_size is not None and int(manifest_size) == staging_size)
    report["size_match"] = size_match
    if not size_match:
        reasons.append("nightly_manifest_size_mismatch")

    manifest_counts = manifest.get("counts") if isinstance(manifest.get("counts"), Mapping) else {}
    staging_counts = table_counts(cfg.staging_db, tuple(manifest_counts.keys())) if staging_exists and manifest_counts else {}
    count_mismatches = {
        key: {"manifest": manifest_counts.get(key), "staging": staging_counts.get(key)}
        for key in manifest_counts
        if staging_counts.get(key) != manifest_counts.get(key)
    }
    report.update(
        {
            "manifest_counts": dict(manifest_counts),
            "staging_counts_for_manifest_keys": staging_counts,
            "count_mismatches": count_mismatches,
        }
    )
    if not manifest_counts:
        reasons.append("nightly_manifest_counts_missing")
    elif count_mismatches:
        reasons.append("nightly_manifest_counts_mismatch")

    service_report, service_report_path, service_report_reasons = _load_nightly_service_report(manifest)
    reasons.extend(service_report_reasons)
    snapshot_manifest = service_report.get("snapshot_manifest") if isinstance(service_report.get("snapshot_manifest"), Mapping) else {}
    failed_required_steps = service_report.get("failed_required_steps")
    failed_required_steps = (
        list(failed_required_steps)
        if isinstance(failed_required_steps, Sequence) and not isinstance(failed_required_steps, (str, bytes))
        else failed_required_steps
    )
    report.update(
        {
            "service_report_path": service_report_path,
            "service_report_run_id": service_report.get("run_id"),
            "latest_published": snapshot_manifest.get("latest_published"),
            "failed_required_steps": failed_required_steps,
            "partial_failure": service_report.get("partial_failure"),
            "overall_status": service_report.get("overall_status"),
        }
    )
    if service_report:
        if snapshot_manifest.get("latest_published") is not True:
            reasons.append("nightly_latest_published_not_true")
        if service_report.get("partial_failure") is not False:
            reasons.append("nightly_partial_failure")
        if failed_required_steps:
            reasons.append("nightly_failed_required_steps")
        if service_report.get("overall_status") != "ok":
            reasons.append("nightly_overall_status_not_ok")
        if service_report.get("run_id") != manifest.get("run_id"):
            reasons.append("nightly_run_id_mismatch")

    report["reasons"] = reasons
    report["ok"] = not reasons
    return report


def _manifest_age_hours(published_at_raw: Any) -> float | None:
    """Hours between now and published_at. Positive means the manifest is
    that old; negative means published_at is in the future. Callers must not
    clamp negative values to zero -- that is what previously let a
    future-dated (clock-skewed or forged) manifest pass as "fresh"; see the
    future-dated fail-closed branch in nightly_manifest_report."""
    text = str(published_at_raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        published_at = datetime.fromisoformat(text)
    except ValueError:
        return None
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - published_at.astimezone(timezone.utc)).total_seconds() / 3600.0


def _load_nightly_service_report(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], str | None, list[str]]:
    raw_path = manifest.get("service_report_path")
    if not raw_path:
        return {}, None, ["nightly_service_report_path_missing"]
    service_report_path = Path(str(raw_path)).expanduser().resolve(strict=False)
    path_str = str(service_report_path)
    if not service_report_path.is_file():
        return {}, path_str, ["nightly_service_report_missing"]
    try:
        payload = json.loads(service_report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, path_str, ["nightly_service_report_unreadable"]
    if not isinstance(payload, Mapping):
        return {}, path_str, ["nightly_service_report_invalid"]
    return dict(payload), path_str, []


def build_report(config_path: Path) -> tuple[dict, bool]:
    cfg = load_config(config_path)
    report = report_base(cfg, "preflight")
    staging = cfg.staging_db
    prod = cfg.prod_db
    snapshot_root = cfg.snapshot_root
    required = max(staging.stat().st_size if staging.exists() else 0, prod.stat().st_size if prod.exists() else 0) * cfg.required_free_copies
    readers = []
    ok = True
    for reader in cfg.readers:
        worktree = Path(str(reader.get("worktree") or "")).expanduser().resolve(strict=False)
        status = git_status_short(worktree) if str(worktree) else None
        classified_status = classify_publish_worktree_status(status)
        reader_report = {
            "name": reader.get("name"),
            "worktree": str(worktree) if str(worktree) else "",
            "git_head": git_head(worktree) if str(worktree) else None,
            "git_status_clean": status == "",
            "clean_for_publish": classified_status["clean_for_publish"],
            "git_status_short": status,
            "tracked_blockers": classified_status["tracked_blockers"],
            "untracked_code_blockers": classified_status["untracked_code_blockers"],
            "untracked_allowed": classified_status["untracked_allowed"],
            "has_stop_command": bool(reader.get("stop_command")),
            "has_start_command": bool(reader.get("start_command")),
            "has_smoke_command_or_internal": bool(reader.get("smoke_command")) or bool(cfg.control_customers),
        }
        if not reader_report["clean_for_publish"]:
            ok = False
        if not reader_report["has_stop_command"] or not reader_report["has_start_command"]:
            ok = False
        readers.append(reader_report)
    diff = schema_diff(prod, staging)
    disk = disk_report(snapshot_root.parent if snapshot_root.parent.exists() else Path.cwd(), required)
    backup = backup_plan_report(
        prod,
        cfg.backup_root,
        cfg.backup_async_copy_root,
        required_bytes=prod.stat().st_size if prod.exists() else 0,
    )
    nightly_manifest = nightly_manifest_report(cfg)
    wal_sidecars = {"prod": wal_sidecar_report(prod), "staging": wal_sidecar_report(staging)}
    report.update(
        {
            "staging_db": str(staging),
            "prod_db": str(prod),
            "snapshot_root": str(snapshot_root),
            "quick_check": {"prod": quick_check(prod), "staging": quick_check(staging)},
            "foreign_key_check": {"prod_rows": len(foreign_key_check(prod)), "staging_rows": len(foreign_key_check(staging))},
            "schema_diff": diff,
            "wal_sidecars": wal_sidecars,
            "counts": {"prod": table_counts(prod, cfg.count_tables), "staging": table_counts(staging, cfg.count_tables)},
            "disk": disk,
            "backup": backup,
            "readers": readers,
            "nightly_manifest": nightly_manifest,
        }
    )
    ok = ok and disk["ok"] and report["quick_check"]["prod"] == "ok" and report["quick_check"]["staging"] == "ok"
    ok = ok and bool(backup["ok"])
    ok = ok and report["foreign_key_check"]["prod_rows"] == 0 and report["foreign_key_check"]["staging_rows"] == 0
    ok = ok and bool(nightly_manifest["ok"])
    ok = ok and bool(diff["ok"])
    ok = ok and bool(wal_sidecars["prod"]["ok"]) and bool(wal_sidecars["staging"]["ok"])
    return report, bool(ok)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight Customer Timeline snapshot publication.")
    add_common_args(parser)
    args = parser.parse_args()
    report, ok = build_report(args.config)
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
