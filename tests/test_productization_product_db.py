from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.product_db import (
    PRODUCT_DB_CAPTURE_INBOX_MIGRATION_ID,
    PRODUCT_DB_MIGRATION_ID,
    PRODUCT_DB_RETENTION_MIGRATION_ID,
    PRODUCT_DB_SCHEDULER_MIGRATION_ID,
    audit_product_db,
    audit_product_retention,
    apply_tenant_owner_config_to_product_db,
    apply_tenant_owner_config_to_product_db_dry_run,
    backup_product_db,
    bootstrap_product_db_from_repository,
    import_repository_snapshot_to_product_db,
    initialize_product_db,
    restore_product_db_from_backup,
    snapshot_tenant_config,
    upgrade_product_db,
)
from mango_mvp.productization.repository import ProductRepository
from scripts import mango_office_product_db_admin
from scripts import mango_office_product_db_bootstrap
from scripts import mango_office_product_owner_config
from tests.test_productization_saas_pass import build_sample_product_db


def test_product_db_initializes_schema_and_migration(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    db_path = product_root / "mango_product_appliance.sqlite"

    report = initialize_product_db(db_path, product_root)

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["migrations_applied"] == 4
    with sqlite3.connect(db_path) as con:
        migrations = {row[0] for row in con.execute("select migration_id from schema_migrations")}
        tables = {
            row[0]
            for row in con.execute("select name from sqlite_master where type = 'table'")
        }
    assert migrations == {
        PRODUCT_DB_MIGRATION_ID,
        PRODUCT_DB_RETENTION_MIGRATION_ID,
        PRODUCT_DB_SCHEDULER_MIGRATION_ID,
        PRODUCT_DB_CAPTURE_INBOX_MIGRATION_ID,
    }
    assert "product_calls" in tables
    assert "tenant_manager_owner_map" in tables
    assert "job_runs" in tables
    assert "capture_inbox_items" in tables
    assert "tenant_config_history" in tables
    assert "retention_policies" in tables


def test_product_db_imports_repository_snapshot_idempotently(tmp_path: Path) -> None:
    source_db, source_root, _audio_dir, _raw_payload = build_sample_product_db(tmp_path)
    source_repo = ProductRepository(source_db, source_root)
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)

    first = import_repository_snapshot_to_product_db(source_repo, product_db, product_root)
    second = import_repository_snapshot_to_product_db(source_repo, product_db, product_root)
    audit = audit_product_db(product_db, product_root)

    assert first["summary"]["validation_ok"] is True
    assert second["summary"]["validation_ok"] is True
    assert audit["summary"]["product_calls"] == 3
    assert audit["summary"]["manager_owner_rows"] == 2
    assert audit["summary"]["calls_with_crm_owner"] == 2
    assert audit["summary"]["pending_owner_mappings"] == 1
    assert audit["summary"]["job_types"] == 5
    assert audit["summary"]["capture_inbox_items"] == 0
    assert audit["summary"]["retention_policies"] == 4
    assert audit["manager_owner_status_counts"] == {"confirmed_candidate": 1, "needs_manual_owner": 1}
    assert audit["call_owner_status_counts"] == {"has_owner": 2, "missing_owner": 1}


def test_product_db_bootstrap_writes_tenant_config_and_audit(tmp_path: Path) -> None:
    source_db, source_root, _audio_dir, _raw_payload = build_sample_product_db(tmp_path)
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    tenant_config = product_root / "config" / "tenant_owner_mapping.json"
    audit_out = product_root / "product_db_bootstrap_audit.json"

    report = bootstrap_product_db_from_repository(
        source_db_path=source_db,
        source_allowed_root=source_root,
        product_db_path=product_db,
        product_root=product_root,
        tenant_owner_config_path=tenant_config,
        replace_existing=True,
        audit_out=audit_out,
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["product_calls"] == 3
    assert report["summary"]["manager_owner_rows"] == 2
    assert report["summary"]["pending_owner_mappings"] == 1
    assert report["integrity"]["summary"]["tenant_config_history"] == 1
    assert report["integrity"]["summary"]["retention_policies"] == 4
    assert tenant_config.exists()
    config = json.loads(tenant_config.read_text(encoding="utf-8"))
    assert config["schema_version"] == "tenant_owner_mapping_v1"
    assert len(config["manager_owner_overrides"]) == 2
    assert audit_out.exists()


def test_product_db_backup_stays_under_product_root(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    db_path = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(db_path, product_root)

    report = backup_product_db(db_path, product_root / "backups" / "backup.sqlite", product_root)

    assert report["validation_ok"] is True
    assert Path(report["backup_path"]).exists()
    with pytest.raises(ValueError, match="backup"):
        backup_product_db(db_path, tmp_path / "outside.sqlite", product_root)


def test_product_db_refuses_runtime_and_unapproved_paths(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    with pytest.raises(ValueError, match="runtime-looking"):
        initialize_product_db(product_root / "mango_mvp.db", product_root)
    with pytest.raises(ValueError, match="stable_runtime"):
        initialize_product_db(tmp_path / "stable_runtime" / "mango_product_appliance.sqlite", tmp_path)
    with pytest.raises(ValueError, match="allowed root"):
        initialize_product_db(tmp_path / "outside" / "mango_product_appliance.sqlite", product_root)


def test_product_db_bootstrap_script_writes_report(tmp_path: Path) -> None:
    source_db, source_root, _audio_dir, _raw_payload = build_sample_product_db(tmp_path)
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    tenant_config = product_root / "config" / "tenant_owner_mapping.json"
    audit_out = product_root / "product_db_bootstrap_audit.json"

    rc = mango_office_product_db_bootstrap.main(
        [
            "--source-root",
            str(source_root),
            "--source-db",
            str(source_db),
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "--tenant-config",
            str(tenant_config),
            "--out",
            str(audit_out),
            "--replace",
        ]
    )

    assert rc == 0
    data = json.loads(audit_out.read_text(encoding="utf-8"))
    assert data["summary"]["validation_ok"] is True
    assert data["summary"]["product_calls"] == 3


def complete_owner_config(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data["manager_owner_overrides"]:
        if entry["manager_extension"] == "102":
            entry["crm_owner_id"] = 9002
            entry["crm_owner_name"] = "Олег Владелец"
            entry["crm_owner_email"] = "oleg.owner@example.com"
            entry["confirmed_by"] = "test"
            entry["notes"] = "manual test mapping"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def bootstrap_sample_product_db(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_db, source_root, _audio_dir, _raw_payload = build_sample_product_db(tmp_path)
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    tenant_config = product_root / "config" / "tenant_owner_mapping.json"
    bootstrap_product_db_from_repository(
        source_db_path=source_db,
        source_allowed_root=source_root,
        product_db_path=product_db,
        product_root=product_root,
        tenant_owner_config_path=tenant_config,
        replace_existing=True,
    )
    return product_root, product_db, tenant_config


def test_owner_config_dry_run_blocks_missing_owner(tmp_path: Path) -> None:
    product_root, product_db, tenant_config = bootstrap_sample_product_db(tmp_path)

    report = apply_tenant_owner_config_to_product_db_dry_run(product_db, tenant_config, product_root)

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked"] == 1
    assert report["summary"]["missing_owner_entries"] == 1
    assert report["summary"]["calls_would_gain_owner"] == 0
    blocked = [action for action in report["actions"] if action["action"].startswith("BLOCK_")]
    assert blocked[0]["action"] == "BLOCK_MISSING_OWNER"
    assert blocked[0]["manager_extension"] == "102"
    assert blocked[0]["calls_affected"] == 1


def test_owner_config_apply_updates_owner_map_and_product_calls(tmp_path: Path) -> None:
    product_root, product_db, tenant_config = bootstrap_sample_product_db(tmp_path)
    complete_owner_config(tenant_config)

    dry_run = apply_tenant_owner_config_to_product_db_dry_run(product_db, tenant_config, product_root)
    report = apply_tenant_owner_config_to_product_db(product_db, tenant_config, product_root)
    audit = audit_product_db(product_db, product_root)

    assert dry_run["summary"]["validation_ok"] is True
    assert dry_run["summary"]["calls_would_gain_owner"] == 1
    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["applied"] == 2
    assert report["summary"]["pending_owner_mappings_after"] == 0
    assert report["summary"]["calls_with_crm_owner_after"] == 3
    assert audit["summary"]["pending_owner_mappings"] == 0
    assert audit["call_owner_status_counts"] == {"has_owner": 3}
    with sqlite3.connect(product_db) as con:
        row = con.execute(
            "select crm_owner_id, crm_owner_name, crm_match_status from product_calls where manager_extension = '102'"
        ).fetchone()
    assert row == (9002, "Олег Владелец", "manual_override")


def test_owner_config_script_writes_dry_run_report(tmp_path: Path) -> None:
    product_root, product_db, tenant_config = bootstrap_sample_product_db(tmp_path)
    out = product_root / "owner_config_audit.json"

    rc = mango_office_product_owner_config.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "--config",
            str(tenant_config),
            "--out",
            str(out),
        ]
    )

    assert rc == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["validation_ok"] is False
    assert data["summary"]["blocked"] == 1


def test_owner_config_script_applies_complete_config(tmp_path: Path) -> None:
    product_root, product_db, tenant_config = bootstrap_sample_product_db(tmp_path)
    complete_owner_config(tenant_config)
    out = product_root / "owner_config_apply_audit.json"

    rc = mango_office_product_owner_config.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "--config",
            str(tenant_config),
            "--out",
            str(out),
            "--apply",
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["applied"] == 2
    assert data["summary"]["pending_owner_mappings_after"] == 0


def test_product_db_upgrade_is_idempotent_and_seeds_retention(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)

    first = upgrade_product_db(product_db, product_root, out_path=product_root / "upgrade.json")
    second = upgrade_product_db(product_db, product_root)
    audit = audit_product_db(product_db, product_root)

    assert first["summary"]["validation_ok"] is True
    assert first["summary"]["migrations_applied"] == 0
    assert second["summary"]["migrations_applied"] == 0
    assert audit["summary"]["schema_migrations"] == 4
    assert audit["summary"]["retention_policies"] == 4
    assert (product_root / "upgrade.json").exists()


def test_snapshot_tenant_config_is_versioned_and_idempotent(tmp_path: Path) -> None:
    product_root, product_db, tenant_config = bootstrap_sample_product_db(tmp_path)

    first = snapshot_tenant_config(product_db, tenant_config, product_root, snapshot_reason="test")
    second = snapshot_tenant_config(product_db, tenant_config, product_root, snapshot_reason="test")
    audit = audit_product_db(product_db, product_root)

    assert first["summary"]["validation_ok"] is True
    assert first["summary"]["already_present"] is True
    assert second["summary"]["already_present"] is True
    assert audit["summary"]["tenant_config_history"] == 1


def test_retention_audit_is_review_only(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    backup_product_db(product_db, product_root / "backups" / "backup.sqlite", product_root)

    report = audit_product_retention(product_db, product_root, out_path=product_root / "retention.json")

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["policies"] == 4
    assert report["summary"]["enabled_policies"] == 3
    assert report["summary"]["artifacts_scanned"] >= 1
    assert report["safety"]["review_only"] is True
    assert (product_root / "retention.json").exists()


def test_restore_product_db_from_backup_replaces_target_safely(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    backup = product_root / "backups" / "restore_source.sqlite"
    backup_product_db(product_db, backup, product_root)
    with sqlite3.connect(product_db) as con:
        con.execute("delete from product_calls")
        con.commit()
    assert audit_product_db(product_db, product_root)["summary"]["product_calls"] == 0

    report = restore_product_db_from_backup(
        backup_path=backup,
        product_db_path=product_db,
        out_allowed_root=product_root,
        replace_existing=True,
        pre_restore_backup_path=product_root / "backups" / "pre_restore.sqlite",
        out_path=product_root / "restore.json",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["integrity"]["summary"]["product_calls"] == 3
    assert Path(report["summary"]["pre_restore_backup_path"]).exists()
    assert (product_root / "restore.json").exists()


def test_product_db_admin_cli_integrity_upgrade_backup_retention_and_snapshot(tmp_path: Path) -> None:
    product_root, product_db, tenant_config = bootstrap_sample_product_db(tmp_path)
    commands = [
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "integrity",
            "--out",
            str(product_root / "integrity.json"),
        ],
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "upgrade",
            "--out",
            str(product_root / "upgrade.json"),
        ],
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "backup",
            "--backup",
            str(product_root / "backups" / "admin.sqlite"),
            "--out",
            str(product_root / "backup.json"),
        ],
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "retention-audit",
            "--out",
            str(product_root / "retention.json"),
        ],
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "snapshot-config",
            "--config",
            str(tenant_config),
            "--out",
            str(product_root / "snapshot.json"),
        ],
    ]

    for argv in commands:
        assert mango_office_product_db_admin.main(argv) == 0

    for name in ("integrity.json", "upgrade.json", "backup.json", "retention.json", "snapshot.json"):
        assert (product_root / name).exists()


def test_product_db_admin_cli_restore(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    backup = product_root / "backups" / "admin_restore.sqlite"
    backup_product_db(product_db, backup, product_root)
    with sqlite3.connect(product_db) as con:
        con.execute("delete from product_calls")
        con.commit()

    rc = mango_office_product_db_admin.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "restore",
            "--backup",
            str(backup),
            "--pre-restore-backup",
            str(product_root / "backups" / "admin_pre_restore.sqlite"),
            "--out",
            str(product_root / "restore.json"),
            "--replace",
        ]
    )

    assert rc == 0
    assert audit_product_db(product_db, product_root)["summary"]["product_calls"] == 3
    assert (product_root / "restore.json").exists()
