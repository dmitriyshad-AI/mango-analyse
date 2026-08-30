#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.publish_snapshot.common import (
    add_common_args,
    backup_plan_report,
    copy_verified,
    finish_cli,
    foreign_key_check,
    git_head,
    lsof_holders,
    load_config,
    remove_sidecars,
    replace_sqlite_verified,
    render_command,
    report_base,
    run_command,
    schema_signature_sha256,
    sha256_file,
    sidecar_paths,
    source_freshness_evidence_ok,
    split_ignored_lsof_holders,
    table_counts,
    user_version,
    wal_checkpoint_truncate,
)
from mango_mvp.customer_timeline.store import (
    customer_timeline_integrity_report,
    customer_timeline_integrity_report_ok,
)
from mango_mvp.customer_timeline.temporal import parse_aware_utc
from scripts.publish_snapshot.reader_smoke import smoke as reader_smoke


def process_pattern_counts(patterns: Sequence[str]) -> list[dict[str, object]]:
    normalized = tuple(str(pattern).strip() for pattern in patterns if str(pattern).strip())
    if not normalized:
        return []
    try:
        proc = subprocess.run(["ps", "-axo", "pid=,command="], text=True, capture_output=True)
    except OSError as exc:
        return [
            {"pattern": pattern, "count": -1, "matches": [], "error": type(exc).__name__}
            for pattern in normalized
        ]
    if proc.returncode != 0:
        return [
            {
                "pattern": pattern,
                "count": -1,
                "matches": [],
                "error": proc.stderr.strip(),
            }
            for pattern in normalized
        ]
    lines = [line.rstrip() for line in proc.stdout.splitlines() if line.strip()]
    return [
        {
            "pattern": pattern,
            "count": sum(1 for line in lines if pattern in line),
            "matches": [line for line in lines if pattern in line][:10],
        }
        for pattern in normalized
    ]


def wait_process_pattern_counts(patterns: Sequence[str], *, timeout: int) -> Mapping[str, object]:
    deadline = time.monotonic() + max(1, timeout)
    last: list[dict[str, object]] = process_pattern_counts(patterns)
    while time.monotonic() <= deadline:
        if last and all(int(item.get("count") or 0) == 1 for item in last):
            return {"ok": True, "checks": last}
        time.sleep(1)
        last = process_pattern_counts(patterns)
    return {"ok": bool(last) and all(int(item.get("count") or 0) == 1 for item in last), "checks": last}


def wait_process_pattern_counts_zero(patterns: Sequence[str], *, timeout: int) -> Mapping[str, object]:
    deadline = time.monotonic() + max(1, timeout)
    last: list[dict[str, object]] = process_pattern_counts(patterns)
    while time.monotonic() <= deadline:
        if all(int(item.get("count") or 0) == 0 for item in last):
            return {"ok": True, "checks": last}
        time.sleep(1)
        last = process_pattern_counts(patterns)
    return {"ok": all(int(item.get("count") or 0) == 0 for item in last), "checks": last}


def _non_ignored_holders(prod_db: Path, cfg) -> tuple[list[str], list[str]]:
    all_holders = lsof_holders(prod_db)
    return split_ignored_lsof_holders(
        all_holders,
        ignored_command_prefixes=tuple(str(item) for item in cfg.raw.get("ignored_lsof_command_prefixes") or ()),
    )


def validate_snapshot_manifest(
    config_path: Path,
    cfg,
    snapshot_db: Path,
    manifest_path: Path,
) -> tuple[dict, bool]:
    try:
        manifest_path = manifest_path.expanduser().resolve(strict=True)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_sha256 = sha256_file(snapshot_db)
        compaction = payload.get("compaction") or {}
        manifest_reader_smoke = payload.get("reader_smoke") or {}
        cutoff = parse_aware_utc(payload.get("cutoff"))
        actual_reader_smoke, actual_reader_ok = reader_smoke(
            config_path,
            snapshot_db=snapshot_db,
            as_of=cutoff,
        ) if cutoff is not None else ({"status": "failed", "reason": "invalid_cutoff"}, False)
        with sqlite3.connect(f"file:{snapshot_db}?mode=ro&immutable=1", uri=True) as con:
            actual_domain_integrity = dict(customer_timeline_integrity_report(con))
        actual_counts = table_counts(snapshot_db, cfg.count_tables)
        expected_headroom = cfg.compact_reader_max_bytes - snapshot_db.stat().st_size
        artifact_under_root = (
            snapshot_db.name == "customer_timeline.sqlite"
            and snapshot_db.parent.parent == cfg.snapshot_root
            and snapshot_db.parent.name.startswith("prod_")
        )
        nightly_manifest = payload.get("nightly_manifest") or {}
        source_freshness = payload.get("source_freshness") or {}
        checks = {
            "artifact_lineage": (
                manifest_path == snapshot_db.parent / "manifest.json"
                and artifact_under_root
                and payload.get("schema_version") == "customer_timeline_snapshot_build_manifest_v3"
                and Path(str(payload.get("snapshot_db") or "")).resolve(strict=False) == snapshot_db
            ),
            "sha256": payload.get("sha256") == actual_sha256,
            "size_bytes": payload.get("size_bytes") == snapshot_db.stat().st_size,
            "user_version": payload.get("user_version") == user_version(snapshot_db),
            "publish_config": payload.get("publish_config_sha256") == sha256_file(config_path),
            "schema": payload.get("schema_sha256") == schema_signature_sha256(snapshot_db),
            "counts": payload.get("counts") == actual_counts and set(actual_counts) == set(cfg.count_tables),
            "build_integrity": (
                payload.get("integrity_check") == payload.get("quick_check") == "ok"
                and payload.get("foreign_key_check_rows") == 0
                and customer_timeline_integrity_report_ok(payload.get("domain_integrity"))
                and actual_reader_smoke.get("quick_check") == "ok"
                and len(foreign_key_check(snapshot_db)) == 0
                and customer_timeline_integrity_report_ok(actual_domain_integrity)
            ),
            "compaction_size": (
                compaction.get("within_size_limit") is True
                and compaction.get("max_size_bytes") == cfg.compact_reader_max_bytes
                and type(compaction.get("size_headroom_bytes")) is int
                and compaction.get("size_headroom_bytes") == expected_headroom
                and expected_headroom >= 0
            ),
            "reader_smoke": (
                manifest_reader_smoke.get("status") == "ok"
                and (manifest_reader_smoke.get("bot_visibility_gate") or {}).get("ok") is True
                and (manifest_reader_smoke.get("bot_visibility_gate") or {}).get("policy") == dict(cfg.bot_visibility_policy)
                and actual_reader_ok
                and actual_reader_smoke.get("bot_visibility") == manifest_reader_smoke.get("bot_visibility")
            ),
            "control_customers": payload.get("control_customers") == list(cfg.control_customers),
            "source_freshness": (
                isinstance(nightly_manifest, Mapping)
                and isinstance(source_freshness, Mapping)
                and source_freshness_evidence_ok(nightly_manifest, source_freshness)
            ),
            "writer_code": payload.get("writer_git_head") == git_head(ROOT),
            "build_lineage": payload.get("writer_identity_stable") is True
            and payload.get("source_unchanged_during_copy") is True
            and payload.get("writer_git_head") == payload.get("writer_git_head_end")
            and nightly_manifest.get("ok") is True
            and not nightly_manifest.get("reasons"),
        }
        return {
            "manifest_path": str(manifest_path),
            "actual_sha256": actual_sha256,
            "expected_sha256": payload.get("sha256"),
            "cutoff": payload.get("cutoff"),
            "checks": checks,
            "actual_reader_smoke": actual_reader_smoke,
        }, all(checks.values())
    except Exception as exc:
        return {"manifest_path": str(manifest_path), "error": type(exc).__name__}, False


def _restore_failed_replacement(backup_db: Path, prod_db: Path, backup_sha256: str) -> tuple[dict, bool]:
    tmp = prod_db.with_suffix(prod_db.suffix + f".auto_rollback_{time.time_ns()}")
    shutil.copy2(backup_db, tmp)
    remove_sidecars(prod_db, execute=True)
    replacement = replace_sqlite_verified(tmp, prod_db)
    ok = bool(replacement.get("ok") and replacement.get("sha256") == backup_sha256)
    return {"attempted": True, "expected_sha256": backup_sha256, "replacement": replacement}, ok


def _rollback_after_runtime_failure(cfg, prod_db: Path, backup_db: Path, backup_sha256: str) -> tuple[dict, bool]:
    stop_results = []
    process_checks = []
    for reader in cfg.readers:
        command = reader.get("stop_command")
        if command:
            worktree = Path(str(reader.get("worktree") or Path.cwd())).expanduser().resolve(strict=False)
            result = run_command(
                render_command(command, {"db": prod_db}),
                cwd=worktree,
                timeout=int(reader.get("stop_timeout_seconds") or 120),
            )
            result["name"] = reader.get("name")
            stop_results.append(result)
        patterns = tuple(str(item) for item in reader.get("process_patterns") or ())
        if patterns:
            check = dict(
                wait_process_pattern_counts_zero(
                    patterns,
                    timeout=int(reader.get("stop_timeout_seconds") or 120),
                )
            )
            check["name"] = reader.get("name")
            process_checks.append(check)
    holders, ignored_holders = _non_ignored_holders(prod_db, cfg)
    safe_to_restore = not holders and all(check.get("ok") is True for check in process_checks)
    if not safe_to_restore:
        return {
            "attempted": False,
            "reason": "reader_not_stopped",
            "stop_results": stop_results,
            "process_checks": process_checks,
            "holders": holders,
            "ignored_holders": ignored_holders,
        }, False
    replacement, ok = _restore_failed_replacement(backup_db, prod_db, backup_sha256)
    return {
        **replacement,
        "stop_results": stop_results,
        "process_checks": process_checks,
        "holders": holders,
        "ignored_holders": ignored_holders,
    }, ok


def flip(
    config_path: Path,
    *,
    snapshot_db: Path,
    snapshot_manifest: Path | None,
    execute: bool,
    restart_readers: bool = False,
) -> tuple[dict, bool]:
    """Atomically swap prod_db to snapshot_db.

    Service restart is a separate, explicit decision from the data swap (see
    Foton/codex_artifacts/ETAP6_publish_plan.md and master TZ 11.6): readers are
    always stopped before the swap (to release locks), but a reader's
    ``start_command`` only runs when the caller passes ``restart_readers=True``.
    Without that flag, flip leaves every reader stopped after a successful swap,
    e.g. to keep the Wappi draft loop off until OWNER-GATE 5A is granted.
    """
    cfg = load_config(config_path)
    report = report_base(cfg, "flip")
    prod_db = cfg.prod_db
    snapshot_db = snapshot_db.expanduser().resolve(strict=True)
    manifest_validation, manifest_ok = (
        validate_snapshot_manifest(config_path, cfg, snapshot_db, snapshot_manifest)
        if snapshot_manifest is not None
        else ({"error": "snapshot_manifest_missing"}, False)
    )
    report.update(
        {
            "execute": execute,
            "restart_readers": restart_readers,
            "prod_db": str(prod_db),
            "snapshot_db": str(snapshot_db),
            "snapshot_manifest_validation": manifest_validation,
        }
    )
    if not execute:
        report["status"] = "dry_run"
        report["release_ready"] = manifest_ok
        return report, True
    if not manifest_ok:
        report["status"] = "blocked_snapshot_manifest"
        return report, False
    backup_check = backup_plan_report(
        prod_db,
        cfg.backup_root,
        cfg.backup_async_copy_root,
        required_bytes=prod_db.stat().st_size,
    )
    if not backup_check["ok"]:
        return {**report, "status": "blocked_backup_preflight", "backup": backup_check}, False

    stop_results = []
    for reader in cfg.readers:
        command = reader.get("stop_command")
        if command:
            worktree = Path(str(reader.get("worktree") or Path.cwd())).expanduser().resolve(strict=False)
            result = run_command(render_command(command, {"db": prod_db}), cwd=worktree, timeout=int(reader.get("stop_timeout_seconds") or 120))
            result["name"] = reader.get("name")
            stop_results.append(result)
    failed_stops = [result for result in stop_results if result.get("rc") != 0]
    if failed_stops:
        return {
            **report,
            "status": "blocked_reader_stop",
            "stop_results": stop_results,
            "failed_stops": failed_stops,
        }, False
    stopped_process_checks = []
    for reader in cfg.readers:
        patterns = tuple(str(item) for item in reader.get("process_patterns") or ())
        if not patterns:
            continue
        check = dict(
            wait_process_pattern_counts_zero(
                patterns,
                timeout=int(reader.get("stop_timeout_seconds") or 120),
            )
        )
        check["name"] = reader.get("name")
        stopped_process_checks.append(check)
    if any(check.get("ok") is not True for check in stopped_process_checks):
        return {
            **report,
            "status": "blocked_reader_process_still_running",
            "stop_results": stop_results,
            "stopped_process_checks": stopped_process_checks,
        }, False
    holders, ignored_holders = _non_ignored_holders(prod_db, cfg)
    if holders:
        return {
            **report,
            "status": "blocked_lsof",
            "stop_results": stop_results,
            "holders": holders,
            "ignored_holders": ignored_holders,
        }, False

    try:
        prod_checkpoint = wal_checkpoint_truncate(prod_db)
        snapshot_sidecars = [str(path) for path in sidecar_paths(snapshot_db) if path.exists()]
        if snapshot_sidecars:
            raise RuntimeError("snapshot sidecars are forbidden")
    except Exception as exc:
        return {
            **report,
            "status": "blocked_checkpoint",
            "stop_results": stop_results,
            "stopped_process_checks": stopped_process_checks,
            "exception": {"type": type(exc).__name__, "message": str(exc), "attempt": 0},
        }, False

    backup_root = cfg.backup_root
    if backup_root is None:
        return {**report, "status": "blocked_backup_root_missing", "backup": backup_check}, False
    async_root = cfg.backup_async_copy_root
    if async_root is None:
        return {**report, "status": "blocked_backup_async_copy_root_missing", "backup": backup_check}, False
    backup_dir = backup_root / ("pre_flip_backup_" + report["generated_at"].replace(":", "").replace("+", "Z"))
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_db = backup_dir / prod_db.name
    backup_copy = copy_verified(prod_db, backup_db)
    backup_sha = str(backup_copy["target_sha256"])
    async_backup_dir = async_root / backup_dir.name
    async_backup_db = async_backup_dir / prod_db.name
    async_backup_copy = copy_verified(backup_db, async_backup_db)
    tmp_target = prod_db.with_suffix(prod_db.suffix + ".new")
    snapshot_copy = copy_verified(snapshot_db, tmp_target)
    expected_snapshot_sha = str(manifest_validation["expected_sha256"])
    if snapshot_copy.get("target_sha256") != expected_snapshot_sha:
        tmp_target.unlink(missing_ok=True)
        return {
            **report,
            "status": "blocked_snapshot_changed_after_manifest_validation",
            "snapshot_copy": snapshot_copy,
            "expected_snapshot_sha256": expected_snapshot_sha,
            "backup_db": str(backup_db),
            "backup_sha256": backup_sha,
        }, False
    pre_replace_holders, pre_replace_ignored_holders = _non_ignored_holders(prod_db, cfg)
    if pre_replace_holders:
        tmp_target.unlink(missing_ok=True)
        return {
            **report,
            "status": "blocked_lsof_before_replace",
            "stop_results": stop_results,
            "holders": pre_replace_holders,
            "ignored_holders": ignored_holders,
            "pre_replace_ignored_lsof_holders": pre_replace_ignored_holders,
            "backup_db": str(backup_db),
            "backup_copy": backup_copy,
            "async_backup_db": str(async_backup_db),
            "async_backup_copy": async_backup_copy,
            "backup_sha256": backup_sha,
            "prod_checkpoint": prod_checkpoint,
            "snapshot_sidecars": snapshot_sidecars,
        }, False
    removed_sidecars = remove_sidecars(prod_db, execute=True)
    replacement = replace_sqlite_verified(tmp_target, prod_db)
    if not replacement["ok"]:
        automatic_rollback = {"attempted": False}
        rollback_ok = replacement.get("replace_completed") is not True
        if replacement.get("replace_completed") is True:
            automatic_rollback, rollback_ok = _restore_failed_replacement(
                backup_db,
                prod_db,
                backup_sha,
            )
        return {
            **report,
            "status": (
                "failed_post_replace_verification_rolled_back"
                if rollback_ok
                else "critical_post_replace_verification_and_rollback_failed"
            ),
            "stop_results": stop_results,
            "backup_db": str(backup_db),
            "backup_copy": backup_copy,
            "async_backup_db": str(async_backup_db),
            "async_backup_copy": async_backup_copy,
            "backup_sha256": backup_sha,
            "prod_checkpoint": prod_checkpoint,
            "snapshot_sidecars": snapshot_sidecars,
            "snapshot_copy": snapshot_copy,
            "removed_sidecars": removed_sidecars,
            "ignored_lsof_holders": ignored_holders,
            "pre_replace_ignored_lsof_holders": pre_replace_ignored_holders,
            "post_replace_verification": replacement,
            "automatic_rollback": automatic_rollback,
        }, False
    new_sha = str(replacement["sha256"])
    installed_reader_smoke, installed_reader_ok = reader_smoke(
        config_path,
        snapshot_db=prod_db,
        as_of=parse_aware_utc(manifest_validation.get("cutoff")),
    )
    ok = bool(new_sha == expected_snapshot_sha and installed_reader_ok)
    if not ok:
        automatic_rollback, rollback_ok = _rollback_after_runtime_failure(
            cfg,
            prod_db,
            backup_db,
            backup_sha,
        )
        return {
            **report,
            "status": (
                "failed_installed_snapshot_validation_rolled_back"
                if rollback_ok
                else "critical_installed_snapshot_validation_and_rollback_failed"
            ),
            "stop_results": stop_results,
            "backup_db": str(backup_db),
            "backup_sha256": backup_sha,
            "new_sha256": new_sha,
            "expected_snapshot_sha256": expected_snapshot_sha,
            "installed_reader_smoke": installed_reader_smoke,
            "automatic_rollback": automatic_rollback,
        }, False

    start_results = []
    post_start_process_checks = []
    post_start_reader_smoke: Mapping[str, object] = {"status": "not_run"}
    skipped_start = []
    for reader in cfg.readers:
        command = reader.get("start_command")
        if not restart_readers:
            if command:
                skipped_start.append({"name": reader.get("name"), "reason": "restart_readers_flag_not_set"})
            continue
        if command:
            worktree = Path(str(reader.get("worktree") or Path.cwd())).expanduser().resolve(strict=False)
            result = run_command(render_command(command, {"db": prod_db}), cwd=worktree, timeout=int(reader.get("start_timeout_seconds") or 120))
            result["name"] = reader.get("name")
            start_results.append(result)
            ok = ok and result.get("rc") == 0
        patterns = tuple(str(item) for item in reader.get("process_patterns") or ())
        if patterns:
            process_check = dict(
                wait_process_pattern_counts(
                    patterns,
                    timeout=int(reader.get("start_timeout_seconds") or 120),
                )
            )
            process_check["name"] = reader.get("name")
            post_start_process_checks.append(process_check)
            ok = ok and bool(process_check.get("ok"))
    if restart_readers and ok:
        post_start_reader_smoke, post_start_reader_ok = reader_smoke(
            config_path,
            snapshot_db=prod_db,
            as_of=parse_aware_utc(manifest_validation.get("cutoff")),
        )
        ok = ok and post_start_reader_ok
    if not ok:
        automatic_rollback, rollback_ok = _rollback_after_runtime_failure(
            cfg,
            prod_db,
            backup_db,
            backup_sha,
        )
        return {
            **report,
            "status": (
                "failed_reader_restart_rolled_back"
                if rollback_ok
                else "critical_reader_restart_and_rollback_failed"
            ),
            "stop_results": stop_results,
            "start_results": start_results,
            "post_start_process_checks": post_start_process_checks,
            "post_start_reader_smoke": post_start_reader_smoke,
            "backup_db": str(backup_db),
            "backup_sha256": backup_sha,
            "new_sha256": new_sha,
            "installed_reader_smoke": installed_reader_smoke,
            "automatic_rollback": automatic_rollback,
        }, False
    report.update(
        {
            "status": "ok",
            "stop_results": stop_results,
            "start_results": start_results,
            "skipped_start": skipped_start,
            "post_start_process_checks": post_start_process_checks,
            "post_start_reader_smoke": post_start_reader_smoke,
            "backup_db": str(backup_db),
            "backup_copy": backup_copy,
            "async_backup_db": str(async_backup_db),
            "async_backup_copy": async_backup_copy,
            "backup_sha256": backup_sha,
            "new_sha256": new_sha,
            "prod_checkpoint": prod_checkpoint,
            "snapshot_sidecars": snapshot_sidecars,
            "snapshot_copy": snapshot_copy,
            "removed_sidecars": removed_sidecars,
            "ignored_lsof_holders": ignored_holders,
            "pre_replace_ignored_lsof_holders": pre_replace_ignored_holders,
            "quick_check": replacement["quick_check"],
            "post_replace_verification": replacement,
            "installed_reader_smoke": installed_reader_smoke,
        }
    )
    return report, True


def main() -> int:
    parser = argparse.ArgumentParser(description="Atomically flip stable Customer Timeline DB path to a built snapshot.")
    add_common_args(parser)
    parser.add_argument("--snapshot-db", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--execute", action="store_true", help="Actually stop readers and replace prod DB.")
    parser.add_argument(
        "--restart-readers",
        action="store_true",
        help=(
            "Restart configured reader services (e.g. Wappi draft loop) after a successful flip. "
            "Default (omitted) stops readers for the swap and leaves them stopped afterward -- pass "
            "this flag only when starting that service is separately approved (e.g. OWNER-GATE 5A)."
        ),
    )
    args = parser.parse_args()
    report, ok = flip(
        args.config,
        snapshot_db=args.snapshot_db,
        snapshot_manifest=args.snapshot_manifest,
        execute=args.execute,
        restart_readers=args.restart_readers,
    )
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
