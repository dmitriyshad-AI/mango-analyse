#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROD_DB = (
    ROOT.parent
    / "Mango analyse"
    / "product_data"
    / "customer_timeline"
    / "customer_timeline_prod_20260621"
    / "customer_timeline.sqlite"
)
DEFAULT_STAGING_DB = ROOT / ".codex_local" / "staging" / "customer_timeline_staging.sqlite"
DEFAULT_CRM_EXPORT = ROOT / ".codex_local" / "staging" / "block7_crm_export_v2_final"
DEFAULT_OUT_DIR = ROOT / ".codex_local" / "transfer_package" / "marathon2_block7_current"

CORE_COUNT_TABLES = (
    "timeline_events",
    "bot_context_chunks",
    "timeline_conflicts",
    "identity_links",
    "customer_identities",
    "customer_opportunities",
    "customer_objections_v1",
    "customer_objection_summary_v1",
    "email_summary_cache_v1",
    "a2v3_mail_event_facts",
    "customer_purchases_v1",
    "derived_signals",
    "family_links_v1",
    "event_child_attribution_v1",
    "opportunity_child_attribution_v1",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Marathon-2 staging transfer/SWAP package.")
    parser.add_argument("--prod-db", type=Path, default=DEFAULT_PROD_DB)
    parser.add_argument("--staging-db", type=Path, default=DEFAULT_STAGING_DB)
    parser.add_argument("--crm-export-dir", type=Path, default=DEFAULT_CRM_EXPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true", help="Replace an existing local transfer package directory.")
    args = parser.parse_args(argv)

    prod_db = args.prod_db.expanduser().resolve(strict=True)
    staging_db = args.staging_db.expanduser().resolve(strict=True)
    crm_export_dir = args.crm_export_dir.expanduser().resolve(strict=True)
    out_dir = args.out_dir.expanduser().resolve(strict=False)
    _guard_paths(prod_db=prod_db, staging_db=staging_db, crm_export_dir=crm_export_dir, out_dir=out_dir)
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"output directory exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    prod_before = _sha256_file(prod_db)
    prod_info = _db_snapshot(prod_db, immutable=True)
    prod_after = _sha256_file(prod_db)
    staging_info = _db_snapshot(staging_db, immutable=True)
    crm_manifest_path = crm_export_dir / "manifest.json"
    crm_manifest = _load_json(crm_manifest_path)
    if str(crm_manifest.get("timeline_db_sha256") or "") != staging_info["sha256"]:
        raise ValueError("CRM export manifest timeline_db_sha256 does not match staging DB sha256.")
    latest_manifest = _load_latest_nightly_manifest()
    diff_rows = _table_diff_rows(prod_info["counts"], staging_info["counts"])
    source_counts = _source_counts(staging_db)
    special_counts = _special_counts(staging_db)

    manifest = {
        "schema_version": "marathon2_block7_transfer_package_v1",
        "generated_at": generated_at,
        "worktree": str(ROOT),
        "branch": _read_git(["branch", "--show-current"]),
        "head": _read_git(["rev-parse", "--short", "HEAD"]),
        "prod_db": str(prod_db),
        "prod_open_mode": "mode=ro&immutable=1; PRAGMA query_only=ON",
        "prod_db_sha256_before": prod_before,
        "prod_db_sha256_after": prod_after,
        "prod_db_untouched": prod_before == prod_after,
        "prod_quick_check": prod_info["quick_check"],
        "prod_foreign_key_check_rows": prod_info["foreign_key_check_rows"],
        "staging_db": str(staging_db),
        "staging_db_sha256": staging_info["sha256"],
        "staging_quick_check": staging_info["quick_check"],
        "staging_foreign_key_check_rows": staging_info["foreign_key_check_rows"],
        "staging_wal_size": _sidecar_size(staging_db, "-wal"),
        "staging_shm_size": _sidecar_size(staging_db, "-shm"),
        "crm_export_package": str(crm_export_dir),
        "crm_export_manifest_path": str(crm_manifest_path),
        "crm_export": {
            "generated_at": crm_manifest.get("generated_at"),
            "candidate_rows": crm_manifest.get("candidate_rows"),
            "pilot_rows": crm_manifest.get("pilot_rows"),
            "ready_rows": crm_manifest.get("ready_rows"),
            "blocked_rows": crm_manifest.get("blocked_rows"),
            "status_counts": crm_manifest.get("status_counts"),
            "blocker_counts": crm_manifest.get("blocker_counts"),
            "safety": crm_manifest.get("safety"),
            "idempotence": crm_manifest.get("idempotence"),
            "timeline_db_sha256": crm_manifest.get("timeline_db_sha256"),
            "output_sha256": crm_manifest.get("output_sha256"),
        },
        "latest_nightly_snapshot": latest_manifest,
        "source_counts": source_counts,
        "special_counts": special_counts,
        "table_counts_prod": prod_info["counts"],
        "table_counts_staging": staging_info["counts"],
        "table_count_diff": diff_rows,
        "hard_safety": {
            "prod_write": False,
            "crm_write": False,
            "amo_write": False,
            "tallanto_write": False,
            "client_sends": False,
            "live_bot_touched": False,
        },
        "known_limits": [
            "CRM package is export-only and requires semantic_pass plus owner approval before any AMO write.",
            "SWAP is an operator package only; Codex did not replace prod DB.",
            "Wappi history remains pending_attribution, not bot-visible memory.",
            "Memory to bot remains shadow-only and default OFF.",
        ],
    }
    (out_dir / "manifest.json").write_text(_json_dumps(manifest), encoding="utf-8")
    (out_dir / "prod_vs_staging_diff.md").write_text(
        _render_diff(generated_at, manifest, diff_rows, source_counts, special_counts),
        encoding="utf-8",
    )
    (out_dir / "crm_package_reference.md").write_text(_render_crm_reference(manifest), encoding="utf-8")
    (out_dir / "swap_apply_scenario.md").write_text(_render_swap_scenario(manifest), encoding="utf-8")
    (out_dir / "rollback_plan.md").write_text(_render_rollback(manifest), encoding="utf-8")
    (out_dir / "README.md").write_text(_render_readme(manifest), encoding="utf-8")
    print(_json_dumps({"out_dir": str(out_dir), "manifest": str(out_dir / "manifest.json"), **manifest["crm_export"]}))
    return 0


def _guard_paths(*, prod_db: Path, staging_db: Path, crm_export_dir: Path, out_dir: Path) -> None:
    if "customer_timeline_prod_20260621" not in prod_db.parts:
        raise ValueError("prod DB path must point to customer_timeline_prod_20260621")
    if ".codex_local" not in staging_db.parts or "staging" not in staging_db.parts:
        raise ValueError("staging DB must be under .codex_local/staging")
    if ".codex_local" not in crm_export_dir.parts or "staging" not in crm_export_dir.parts:
        raise ValueError("CRM export must be under .codex_local/staging")
    if ".codex_local" not in out_dir.parts or "transfer_package" not in out_dir.parts:
        raise ValueError("transfer package output must be under .codex_local/transfer_package")


def _db_snapshot(db_path: Path, *, immutable: bool) -> dict[str, Any]:
    con = _connect_ro(db_path, immutable=immutable)
    try:
        quick = str(con.execute("PRAGMA quick_check").fetchone()[0])
        fk_rows = len(con.execute("PRAGMA foreign_key_check").fetchall())
        counts = {table: _count_table(con, table) for table in CORE_COUNT_TABLES}
        counts.update(_all_extra_table_counts(con, set(counts)))
        return {
            "sha256": _sha256_file(db_path),
            "quick_check": quick,
            "foreign_key_check_rows": fk_rows,
            "counts": counts,
        }
    finally:
        con.close()


def _connect_ro(db_path: Path, *, immutable: bool) -> sqlite3.Connection:
    query = "mode=ro&immutable=1" if immutable else "mode=ro"
    con = sqlite3.connect(f"file:{db_path}?{query}", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    return con


def _all_extra_table_counts(con: sqlite3.Connection, existing: set[str]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for row in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
        name = str(row["name"])
        if name in existing:
            continue
        if name.startswith("sqlite_"):
            continue
        if name.endswith(("_data", "_idx", "_docsize", "_content", "_config", "_keys")):
            continue
        result[name] = _count_table(con, name)
    return result


def _count_table(con: sqlite3.Connection, table: str) -> int | None:
    if not _table_exists(con, table):
        return None
    return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _source_counts(db_path: Path) -> list[dict[str, Any]]:
    con = _connect_ro(db_path, immutable=True)
    try:
        return [
            dict(row)
            for row in con.execute(
                """
                SELECT source_system, event_type, COUNT(*) AS rows, COUNT(DISTINCT customer_id) AS customers,
                       MIN(event_at) AS min_event_at, MAX(event_at) AS max_event_at
                FROM timeline_events
                GROUP BY source_system, event_type
                ORDER BY rows DESC, source_system, event_type
                """
            ).fetchall()
        ]
    finally:
        con.close()


def _special_counts(db_path: Path) -> dict[str, Any]:
    con = _connect_ro(db_path, immutable=True)
    try:
        return {
            "mail_stage2_manager_only_chunks": _scalar(
                con,
                "SELECT COUNT(*) FROM bot_context_chunks WHERE source_system='mail_archive_stage2' AND allowed_for_bot=0 AND requires_manager_review=1",
            ),
            "mail_stage2_allowed_chunks": _scalar(
                con,
                "SELECT COUNT(*) FROM bot_context_chunks WHERE source_system='mail_archive_stage2' AND allowed_for_bot=1",
            ),
            "wappi_timeline_events": _scalar(
                con,
                "SELECT COUNT(*) FROM timeline_events WHERE source_system IN ('wappi_telegram','wappi_max')",
            ),
            "wappi_pending_attribution": _scalar(
                con,
                "SELECT COUNT(*) FROM timeline_conflicts WHERE conflict_type='pending_attribution' AND (record_json LIKE '%wappi_telegram%' OR record_json LIKE '%wappi_max%')",
            ),
            "money_fact_rows": _scalar(
                con,
                "SELECT COUNT(*) FROM customer_purchases_v1 WHERE money_kind='fact'",
            ),
            "money_plan_rows": _scalar(
                con,
                "SELECT COUNT(*) FROM customer_purchases_v1 WHERE money_kind='plan'",
            ),
            "active_signals": _scalar(
                con,
                "SELECT COUNT(*) FROM derived_signals WHERE COALESCE(status,'active')='active'",
            ),
            "family_links": _scalar(con, "SELECT COUNT(*) FROM family_links_v1"),
        }
    finally:
        con.close()


def _scalar(con: sqlite3.Connection, sql: str) -> int:
    try:
        return int(con.execute(sql).fetchone()[0] or 0)
    except sqlite3.Error:
        return 0


def _table_diff_rows(prod: Mapping[str, int | None], staging: Mapping[str, int | None]) -> list[dict[str, Any]]:
    rows = []
    for table in sorted(set(prod) | set(staging)):
        before = prod.get(table)
        after = staging.get(table)
        rows.append({"table": table, "prod": before, "staging": after, "delta": None if before is None or after is None else after - before})
    return rows


def _render_diff(
    generated_at: str,
    manifest: Mapping[str, Any],
    diff_rows: Iterable[Mapping[str, Any]],
    source_counts: Iterable[Mapping[str, Any]],
    special_counts: Mapping[str, Any],
) -> str:
    lines = [
        "# Marathon-2 prod vs staging diff",
        "",
        f"Generated: `{generated_at}`",
        "",
        "## Safety",
        "",
        f"- Prod sha before: `{manifest['prod_db_sha256_before']}`",
        f"- Prod sha after: `{manifest['prod_db_sha256_after']}`",
        f"- Prod untouched: `{manifest['prod_db_untouched']}`",
        f"- Prod open mode: `{manifest['prod_open_mode']}`",
        f"- Prod quick_check: `{manifest['prod_quick_check']}`",
        f"- Staging sha: `{manifest['staging_db_sha256']}`",
        f"- Staging quick_check: `{manifest['staging_quick_check']}`",
        "",
        "## Special counts",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in special_counts.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Source counts", "", "| source_system | event_type | rows | customers | min_event_at | max_event_at |", "|---|---|---:|---:|---|---|"])
    for row in source_counts:
        lines.append(
            f"| `{row['source_system']}` | `{row['event_type']}` | `{row['rows']}` | `{row['customers']}` | `{row['min_event_at'] or ''}` | `{row['max_event_at'] or ''}` |"
        )
    lines.extend(["", "## Table counts", "", "| table | prod | staging | delta |", "|---|---:|---:|---:|"])
    for row in diff_rows:
        lines.append(f"| `{row['table']}` | `{row['prod']}` | `{row['staging']}` | `{row['delta']}` |")
    return "\n".join(lines).rstrip() + "\n"


def _render_crm_reference(manifest: Mapping[str, Any]) -> str:
    crm = manifest["crm_export"]
    return f"""# CRM package reference

Local CRM export package: `{manifest['crm_export_package']}`

Rows:

- candidates: `{crm['candidate_rows']}`
- pilot: `{crm['pilot_rows']}`
- ready: `{crm['ready_rows']}`
- blocked: `{crm['blocked_rows']}`
- source DB sha: `{crm['timeline_db_sha256']}`
- idempotence: `{crm.get('idempotence', {}).get('passed')}`

Safety:

```json
{json.dumps(crm.get('safety') or {}, ensure_ascii=False, indent=2, sort_keys=True)}
```

This package is export-only. AMO live-write requires separate semantic_pass,
owner approval, D7 anti-clobber, staged write and readback.
"""


def _render_swap_scenario(manifest: Mapping[str, Any]) -> str:
    return f"""# SWAP apply scenario — do not execute from Codex

This is an operator checklist. Codex did not replace prod DB.

## Inputs

- Prod DB: `{manifest['prod_db']}`
- Expected prod sha before apply: `{manifest['prod_db_sha256_before']}`
- Staging DB: `{manifest['staging_db']}`
- Staging sha: `{manifest['staging_db_sha256']}`
- Staging WAL size at package build: `{manifest['staging_wal_size']}`
- Staging SHM size at package build: `{manifest['staging_shm_size']}`

## Steps

1. Freeze confirmation: verify nobody wrote to prod since package generation.
2. On STAGING: run `PRAGMA wal_checkpoint(TRUNCATE)` immediately before copying; if any writer ran after checkpoint, repeat it. Confirm `-wal` size is `0`.
3. Stop the live bot and any customer_timeline writer. Wait until no process holds the prod DB or writer lock.
4. Backup prod: first check free disk space (`df -h`, at least 2x DB size), copy prod `.sqlite`, and write backup sha256. Record owner/mode with `stat`.
5. Replace prod DB: copy staging `.sqlite` to prod location. Do not copy `-wal`, `-shm`, `-journal`, or writer locks. Remove old prod sidecars. Restore owner/mode from step 4.
6. Smoke read-only: `PRAGMA quick_check`, FK-check, table counts equal this manifest, and 3 control customers read through CustomerTimelineReadApi.
7. Start the bot. Run one controlled smoke dialogue. Keep memory flags as previously approved; Marathon-2 does not enable live memory.
8. Rollback if any smoke/check fails: stop bot, restore backup `.sqlite`, remove sidecars, smoke again, start bot.

## CRM package

CRM package application is separate from DB SWAP. Use D7 writeback only after semantic_pass and owner approval.
"""


def _render_rollback(manifest: Mapping[str, Any]) -> str:
    return f"""# Rollback plan

Rollback applies only if owner later applies the SWAP package.

1. Keep the pre-apply prod backup until owner accepts the new DB.
2. If quick_check, FK-check, count smoke, read-api smoke, or bot start fails:
   - stop bot/writers;
   - restore the backup `.sqlite`;
   - remove prod `-wal`, `-shm`, `-journal`, and writer locks;
   - run smoke checks again;
   - restart bot.
3. CRM writeback rollback is not automatic in this package. Use D7 snapshot/journal rollback and readback.

Expected pre-apply prod sha: `{manifest['prod_db_sha256_before']}`
Staging sha prepared for apply: `{manifest['staging_db_sha256']}`
"""


def _render_readme(manifest: Mapping[str, Any]) -> str:
    return f"""# Marathon-2 Block 7 transfer package

Generated: `{manifest['generated_at']}`

This package prepares the owner-facing transfer artifacts for Marathon-2.
It does not write to prod, AMO, Tallanto, CRM, stable_runtime, or live bot.

Files:

- `manifest.json` — package source of truth;
- `prod_vs_staging_diff.md` — count diff and source inventory;
- `crm_package_reference.md` — current CRM export package pointer;
- `swap_apply_scenario.md` — owner SWAP checklist;
- `rollback_plan.md` — rollback checklist.

Current CRM export:

- candidates: `{manifest['crm_export']['candidate_rows']}`
- ready: `{manifest['crm_export']['ready_rows']}`
- blocked: `{manifest['crm_export']['blocked_rows']}`

Known constraints:

{chr(10).join(f'- {item}' for item in manifest['known_limits'])}
"""


def _load_latest_nightly_manifest() -> Mapping[str, Any]:
    path = ROOT / ".codex_local" / "staging" / "nightly_service" / "published" / "latest_customer_timeline_snapshot.json"
    if not path.exists():
        return {}
    return _load_json(path)


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar_size(db_path: Path, suffix: str) -> int:
    path = Path(str(db_path) + suffix)
    return path.stat().st_size if path.exists() else 0


def _read_git(args: list[str]) -> str:
    import subprocess

    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
