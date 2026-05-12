from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.product_db import initialize_product_db
from mango_mvp.productization.product_ops import (
    build_product_ops_diagnostics_bundle,
    build_product_ops_healthcheck,
    build_restore_dry_run,
    run_product_db_backup,
    sqlite_quick_check,
    verify_product_db_backup,
)
from scripts import mango_office_product_ops


def test_product_ops_healthcheck_backup_verify_restore_dry_run(tmp_path: Path) -> None:
    product_root = tmp_path / "product appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    backup_path = product_root / "backups" / "mango_product_appliance_backup.sqlite"

    health = build_product_ops_healthcheck(product_root, product_db, product_root / "ops" / "healthcheck.json")
    backup = run_product_db_backup(product_root, product_db, backup_path, product_root / "ops" / "backup.json")
    verify = verify_product_db_backup(product_root, product_db, backup_path, product_root / "ops" / "verify.json")
    restore = build_restore_dry_run(product_root, product_db, backup_path, product_root / "ops" / "restore_dry_run.json")

    assert health["summary"]["validation_ok"] is True
    assert backup["summary"]["validation_ok"] is True
    assert verify["summary"]["validation_ok"] is True
    assert restore["summary"]["execute_restore"] is False
    assert restore["safety"]["restore_executed"] is False
    assert sqlite_quick_check(backup_path) == "ok"
    assert backup_path.exists()


def test_product_ops_diagnostics_bundle_writes_manifest_without_backup_or_restore(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    out_dir = product_root / "ops" / "diagnostics"

    report = build_product_ops_diagnostics_bundle(product_root, product_db, out_dir=out_dir)

    assert report["summary"]["operation"] == "diagnostics_bundle"
    assert report["summary"]["artifacts"] == 4
    assert report["summary"]["backup_executed"] is False
    assert report["summary"]["restore_executed"] is False
    assert (out_dir / "diagnostics_manifest.json").exists()
    assert (out_dir / "healthcheck.json").exists()
    assert report["safety"]["restore_executed"] is False


def test_product_ops_cli_commands_write_reports(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    backup_path = product_root / "backups" / "backup.sqlite"
    health_out = product_root / "ops" / "healthcheck.json"
    backup_out = product_root / "ops" / "backup.json"
    verify_out = product_root / "ops" / "verify.json"
    restore_out = product_root / "ops" / "restore.json"

    health_rc = mango_office_product_ops.main(
        ["--product-root", str(product_root), "--product-db", str(product_db), "healthcheck", "--out", str(health_out)]
    )
    backup_rc = mango_office_product_ops.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "backup",
            "--backup",
            str(backup_path),
            "--out",
            str(backup_out),
        ]
    )
    verify_rc = mango_office_product_ops.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "verify-backup",
            "--backup",
            str(backup_path),
            "--out",
            str(verify_out),
        ]
    )
    restore_rc = mango_office_product_ops.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "restore-dry-run",
            "--backup",
            str(backup_path),
            "--out",
            str(restore_out),
        ]
    )
    diagnostics_rc = mango_office_product_ops.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "diagnostics",
            "--out-dir",
            str(product_root / "ops" / "diagnostics"),
        ]
    )

    assert (health_rc, backup_rc, verify_rc, restore_rc, diagnostics_rc) == (0, 0, 0, 0, 0)
    assert json.loads(restore_out.read_text(encoding="utf-8"))["summary"]["execute_restore"] is False
    assert (product_root / "ops" / "diagnostics" / "diagnostics_manifest.json").exists()


def test_product_ops_refuses_paths_outside_product_root(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)

    with pytest.raises(ValueError, match="backup"):
        run_product_db_backup(product_root, product_db, tmp_path / "outside.sqlite")
