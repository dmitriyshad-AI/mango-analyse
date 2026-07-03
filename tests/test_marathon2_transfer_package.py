from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts import build_marathon2_transfer_package as transfer


def _init_db(path: Path, *, events: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE timeline_events (
            event_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            source_system TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_at TEXT NOT NULL
        );
        CREATE TABLE bot_context_chunks (
            chunk_id TEXT PRIMARY KEY,
            source_system TEXT NOT NULL,
            allowed_for_bot INTEGER NOT NULL DEFAULT 0,
            requires_manager_review INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE timeline_conflicts (
            conflict_id TEXT PRIMARY KEY,
            conflict_type TEXT NOT NULL,
            record_json TEXT NOT NULL
        );
        CREATE TABLE customer_purchases_v1 (
            tenant_id TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            period TEXT NOT NULL,
            money_kind TEXT NOT NULL,
            total_in REAL,
            total_out REAL,
            deals_cnt INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (tenant_id, customer_id, period, money_kind)
        );
        CREATE TABLE derived_signals (
            signal_id TEXT PRIMARY KEY,
            status TEXT
        );
        CREATE TABLE family_links_v1 (
            family_link_id TEXT PRIMARY KEY
        );
        """
    )
    for index in range(events):
        con.execute(
            "INSERT INTO timeline_events VALUES (?, 'foton', 'customer:1', 'mail_archive_stage2', 'email_message', ?)",
            (f"event:{index}", f"2026-07-03T00:00:0{index}+00:00"),
        )
    con.execute("INSERT INTO bot_context_chunks VALUES ('chunk:1', 'mail_archive_stage2', 0, 1)")
    con.execute("INSERT INTO timeline_conflicts VALUES ('conflict:1', 'pending_attribution', '{\"source_system\":\"wappi_max\"}')")
    con.execute("INSERT INTO customer_purchases_v1 VALUES ('foton', 'customer:1', 'all', 'fact', 100, 0, 1)")
    con.execute("INSERT INTO derived_signals VALUES ('signal:1', 'active')")
    con.execute("INSERT INTO family_links_v1 VALUES ('family:1')")
    con.commit()
    con.close()


def _write_crm_manifest(path: Path, *, db_sha: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-07-03T10:00:00+00:00",
                "candidate_rows": 1,
                "pilot_rows": 1,
                "ready_rows": 1,
                "blocked_rows": 0,
                "status_counts": {"да": 1},
                "blocker_counts": {},
                "safety": {"write_amo": False, "write_tallanto": False, "send_messages": False, "prod_db_write": False},
                "idempotence": {"checked": True, "passed": True},
                "timeline_db_sha256": db_sha,
                "output_sha256": {"batch_jsonl": "abc"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_transfer_package_builds_local_swap_manifest(tmp_path: Path) -> None:
    prod = tmp_path / "Mango analyse" / "product_data" / "customer_timeline" / "customer_timeline_prod_20260621" / "customer_timeline.sqlite"
    staging = tmp_path / "work" / ".codex_local" / "staging" / "customer_timeline_staging.sqlite"
    crm = tmp_path / "work" / ".codex_local" / "staging" / "crm"
    out = tmp_path / "work" / ".codex_local" / "transfer_package" / "pkg"
    _init_db(prod, events=1)
    _init_db(staging, events=2)
    staging_sha = transfer._sha256_file(staging)
    prod_sha = transfer._sha256_file(prod)
    _write_crm_manifest(crm, db_sha=staging_sha)

    rc = transfer.main(["--prod-db", str(prod), "--staging-db", str(staging), "--crm-export-dir", str(crm), "--out-dir", str(out)])

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prod_db_sha256_before"] == prod_sha
    assert manifest["prod_db_sha256_after"] == prod_sha
    assert manifest["prod_db_untouched"] is True
    assert manifest["staging_db_sha256"] == staging_sha
    assert manifest["crm_export"]["ready_rows"] == 1
    assert (out / "swap_apply_scenario.md").exists()
    assert (out / "rollback_plan.md").exists()


def test_transfer_package_rejects_crm_manifest_for_other_db(tmp_path: Path) -> None:
    prod = tmp_path / "Mango analyse" / "product_data" / "customer_timeline" / "customer_timeline_prod_20260621" / "customer_timeline.sqlite"
    staging = tmp_path / "work" / ".codex_local" / "staging" / "customer_timeline_staging.sqlite"
    crm = tmp_path / "work" / ".codex_local" / "staging" / "crm"
    out = tmp_path / "work" / ".codex_local" / "transfer_package" / "pkg"
    _init_db(prod)
    _init_db(staging)
    _write_crm_manifest(crm, db_sha="wrong")

    with pytest.raises(ValueError, match="timeline_db_sha256"):
        transfer.main(["--prod-db", str(prod), "--staging-db", str(staging), "--crm-export-dir", str(crm), "--out-dir", str(out)])


def test_transfer_package_rejects_non_local_output(tmp_path: Path) -> None:
    prod = tmp_path / "Mango analyse" / "product_data" / "customer_timeline" / "customer_timeline_prod_20260621" / "customer_timeline.sqlite"
    staging = tmp_path / "work" / ".codex_local" / "staging" / "customer_timeline_staging.sqlite"
    crm = tmp_path / "work" / ".codex_local" / "staging" / "crm"
    _init_db(prod)
    _init_db(staging)
    _write_crm_manifest(crm, db_sha=transfer._sha256_file(staging))

    with pytest.raises(ValueError, match="transfer package output"):
        transfer.main(["--prod-db", str(prod), "--staging-db", str(staging), "--crm-export-dir", str(crm), "--out-dir", str(tmp_path / "bad")])
