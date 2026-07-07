from __future__ import annotations

from contextlib import contextmanager
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import mango_mvp.customer_timeline.nightly_service as nightly_service_module
from mango_mvp.customer_timeline import CustomerIdentity, CustomerTimelineSQLiteStore, IdentityStatus
from mango_mvp.customer_timeline.nightly_service import run_nightly_service, service_config_from_json


NOW = datetime(2026, 7, 3, 3, 20, tzinfo=timezone.utc)


def seed_customer(db_path: Path, allowed_root: Path) -> None:
    with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:nightly-1",
                identity_status=IdentityStatus.STRONG,
                display_name="Тестовый клиент",
                first_seen_at=NOW,
                last_seen_at=NOW,
                touch_count=1,
                created_at=NOW,
                updated_at=NOW,
            )
        )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_service_config(tmp_path: Path, *, enabled: bool = True) -> Path:
    db_path = tmp_path / "customer_timeline.sqlite"
    source_path = tmp_path / "source.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "source_id": "nightly-event-1",
                "customer_id": "customer:nightly-1",
                "event_type": "system_note",
                "event_at": "2026-07-03T03:00:00+00:00",
                "updated_at": "2026-07-03T03:00:00+00:00",
                "direction": "system",
                "summary": "Ночной тестовый импорт.",
            }
        ],
    )
    config = {
        "timeline_db": str(db_path),
        "allowed_root": str(tmp_path),
        "out_root": str(tmp_path / "nightly_service"),
        "publish_dir": str(tmp_path / "published"),
        "tenant_id": "foton",
        "steps": [
            {
                "name": "local_jsonl",
                "kind": "nightly_incremental",
                "enabled": enabled,
                "config": {
                    "journal_path": str(tmp_path / "nightly_service" / "journal.jsonl"),
                    "safety_margin_seconds": 60,
                    "sources": [
                        {
                            "name": "local_jsonl",
                            "source_system": "nightly_test_source",
                            "path": str(source_path),
                            "source_ref": "test:nightly",
                            "normalizer": "jsonl",
                        }
                    ],
                },
            }
        ],
    }
    path = tmp_path / "service_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_nightly_service_publishes_manifest_and_second_run_has_no_changes(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    config_path = write_service_config(tmp_path)
    config = service_config_from_json(config_path)

    first = run_nightly_service(config)
    second = run_nightly_service(config)

    assert first["steps"][0]["status"] == "ok"
    assert first["steps"][0]["summary"]["changed_customer_count"] == 1
    assert second["steps"][0]["summary"]["changed_customer_count"] == 0
    assert first["snapshot_manifest"]["counts"]["timeline_events"] == 1
    latest = tmp_path / "published" / "latest_customer_timeline_snapshot.json"
    assert latest.exists()
    manifest = json.loads(latest.read_text(encoding="utf-8"))
    assert manifest["quick_check"] == "ok"
    assert manifest["files"]["sqlite"]["exists"] is True
    assert manifest["files"]["sqlite"]["sha256"]


def test_nightly_service_keeps_service_lock_through_manifest_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    config = service_config_from_json(write_service_config(tmp_path))
    state = {"locked": False}
    original_manifest = nightly_service_module.build_snapshot_manifest

    @contextmanager
    def observed_lock(*args, **kwargs):
        state["locked"] = True
        try:
            yield {"path": str(tmp_path / "observed.lock"), "waited_seconds": 0.0}
        finally:
            state["locked"] = False

    def observed_manifest(*args, **kwargs):
        assert state["locked"] is True
        return original_manifest(*args, **kwargs)

    monkeypatch.setattr(nightly_service_module, "service_lock", observed_lock)
    monkeypatch.setattr(nightly_service_module, "build_snapshot_manifest", observed_manifest)

    report = run_nightly_service(config)

    assert report["snapshot_manifest"]["counts"]["timeline_events"] == 1
    assert state["locked"] is False


def test_nightly_service_rejects_paths_outside_allowed_root(tmp_path: Path) -> None:
    seed_customer(tmp_path / "customer_timeline.sqlite", tmp_path)
    config_path = write_service_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["publish_dir"] = str(tmp_path.parent / "outside-published")
    config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="allowed root"):
        service_config_from_json(config_path)


def test_nightly_service_rejects_prod_timeline_path(tmp_path: Path) -> None:
    config_path = write_service_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    prod_root = tmp_path / "customer_timeline_prod_20260621"
    payload["allowed_root"] = str(tmp_path)
    payload["timeline_db"] = str(prod_root / "customer_timeline.sqlite")
    config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="prod DB"):
        service_config_from_json(config_path)


def test_nightly_service_local_freshness_monitor_is_optional_and_writes_no_events(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT INTO ingestion_cursors (tenant_id, source_system, last_cursor_ts, updated_at, metadata_json)
            VALUES ('foton', 'wappi_history', '2026-06-01T00:00:00+00:00', '2026-06-01T00:00:00+00:00', '{}')
            """
        )
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"pending_attribution": 3}, ensure_ascii=False), encoding="utf-8")
    config_payload = {
        "timeline_db": str(db_path),
        "allowed_root": str(tmp_path),
        "out_root": str(tmp_path / "nightly_service"),
        "publish_dir": str(tmp_path / "published"),
        "tenant_id": "foton",
        "steps": [
            {
                "name": "wappi_history_incremental",
                "kind": "local_freshness_monitor",
                "enabled": True,
                "required": False,
                "config": {
                    "metrics_path": str(metrics_path),
                    "paths": [str(metrics_path)],
                    "cursor_source_system": "wappi_history_pending",
                    "cursor_ts": "2026-06-21T10:00:00+00:00",
                    "deprecated_cursor_source_systems": ["wappi_history"],
                    "reason": "pending_only",
                },
            }
        ],
    }
    config_path = tmp_path / "monitor_service_config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["overall_status"] == "ok"
    assert report["steps"][0]["status"] == "ok"
    assert report["steps"][0]["summary"]["metrics"]["pending_attribution"] == 3
    assert report["steps"][0]["summary"]["deprecated_cursors_removed"] == ["wappi_history"]
    assert report["snapshot_manifest"]["counts"]["timeline_events"] == 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        cursors = con.execute("SELECT source_system, last_cursor_ts FROM ingestion_cursors ORDER BY source_system").fetchall()
    assert cursors == [("wappi_history_pending", "2026-06-21T10:00:00+00:00")]


def test_nightly_service_runs_optional_mail_link_enrich_and_publishes_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    config_payload = {
        "timeline_db": str(db_path),
        "allowed_root": str(tmp_path),
        "out_root": str(tmp_path / "nightly_service"),
        "publish_dir": str(tmp_path / "published"),
        "tenant_id": "foton",
        "steps": [
            {
                "name": "mail_link_enrich",
                "kind": "mail_link_enrich",
                "enabled": True,
                "required": False,
                "config": {
                    "timeline_db": str(db_path),
                    "allowed_root": str(tmp_path),
                    "out_dir": str(tmp_path / "mail_link_enrich"),
                    "apply": True,
                },
            }
        ],
    }
    config_path = tmp_path / "mail_link_service_config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["overall_status"] == "ok"
    assert report["steps"][0]["status"] == "ok"
    assert report["steps"][0]["summary"]["target_events"] == 0
    manifest = json.loads((tmp_path / "published" / "latest_customer_timeline_snapshot.json").read_text(encoding="utf-8"))
    assert manifest["mail_link_enrich"]["status"] == "ok"
    assert manifest["mail_link_enrich"]["linked_strong"] == 0
    assert manifest["source_counts"] == []


def test_nightly_service_imports_mango_processed_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    source_path = tmp_path / "mango_calls.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "call_id": "call-nightly-1",
                "source_id": "call-nightly-1",
                "customer_id": "customer:nightly-1",
                "identity_authority": "existing_timeline_increment",
                "identity_resolved_by_increment": True,
                "match_class": "strong_unique",
                "started_at": "2026-07-03T02:55:00+00:00",
                "updated_at": "2026-07-03T02:56:00+00:00",
                "direction": "inbound",
                "summary": "Клиент уточнил стоимость и сроки.",
                "allowed_for_bot": False,
            }
        ],
    )
    config_payload = {
        "timeline_db": str(db_path),
        "allowed_root": str(tmp_path),
        "out_root": str(tmp_path / "nightly_service"),
        "publish_dir": str(tmp_path / "published"),
        "tenant_id": "foton",
        "steps": [
            {
                "name": "mango_calls",
                "kind": "nightly_incremental",
                "enabled": True,
                "config": {
                    "journal_path": str(tmp_path / "nightly_service" / "journal.jsonl"),
                    "safety_margin_seconds": 0,
                    "sources": [
                        {
                            "name": "mango_calls",
                            "source_system": "mango_processed_summary",
                            "path": str(source_path),
                            "source_ref": "mango:nightly-test",
                            "normalizer": "mango_processed_summary",
                        }
                    ],
                },
            }
        ],
    }
    config_path = tmp_path / "mango_service_config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    config = service_config_from_json(config_path)

    first = run_nightly_service(config)
    second = run_nightly_service(config)

    assert first["steps"][0]["summary"]["changed_customer_count"] == 1
    assert second["steps"][0]["summary"]["changed_customer_count"] == 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        row = con.execute(
            "SELECT event_id, event_type, source_system FROM timeline_events WHERE source_id = ?",
            ("call-nightly-1",),
        ).fetchone()
        chunk_row = con.execute(
            "SELECT allowed_for_bot, requires_manager_review FROM bot_context_chunks WHERE event_id = ?",
            (row[0],),
        ).fetchone()
    assert row[1:] == ("mango_call", "mango_processed_summary")
    assert chunk_row == (0, 1)


def test_nightly_service_fail_closes_mango_processed_summary_allowed_for_bot_true(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    source_path = tmp_path / "mango_calls_unsafe.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "call_id": "call-nightly-unsafe",
                "customer_id": "customer:nightly-1",
                "identity_authority": "existing_timeline_increment",
                "identity_resolved_by_increment": True,
                "match_class": "strong_unique",
                "started_at": "2026-07-03T02:55:00+00:00",
                "updated_at": "2026-07-03T02:56:00+00:00",
                "summary": "Клиент уточнил стоимость.",
                "allowed_for_bot": True,
            }
        ],
    )
    payload = json.loads(write_service_config(tmp_path).read_text(encoding="utf-8"))
    payload["steps"][0]["config"]["sources"][0] = {
        "name": "mango_calls",
        "source_system": "mango_processed_summary",
        "path": str(source_path),
        "source_ref": "mango:nightly-test",
        "normalizer": "mango_processed_summary",
    }
    config_path = tmp_path / "unsafe_mango_service_config.json"
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    config = service_config_from_json(config_path)

    report = run_nightly_service(config)

    assert report["overall_status"] == "partial"
    assert report["steps"][0]["status"] == "failed_required_source"
    assert report["steps"][0]["summary"]["failed_required_sources"] == ["mango_calls"]
    assert report["snapshot_manifest"]["latest_published"] is False
    assert not (tmp_path / "published" / "latest_customer_timeline_snapshot.json").exists()
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        assert con.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0] == 0


def test_nightly_service_optional_source_failure_keeps_latest_publish(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    config_path = write_service_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["steps"][0]["config"]["sources"].append(
        {
            "name": "optional_missing",
            "source_system": "optional_missing_source",
            "path": str(tmp_path / "missing_optional.jsonl"),
            "source_ref": "test:optional-missing",
            "normalizer": "jsonl",
            "required": False,
        }
    )
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    config = service_config_from_json(config_path)

    report = run_nightly_service(config)

    assert report["overall_status"] == "ok"
    assert report["steps"][0]["status"] == "ok"
    assert report["steps"][0]["summary"]["source_statuses"] == {"ok": 1, "skipped": 1}
    assert report["snapshot_manifest"]["latest_published"] is True
    assert (tmp_path / "published" / "latest_customer_timeline_snapshot.json").exists()


def test_nightly_service_records_disabled_step_without_running(tmp_path: Path) -> None:
    seed_customer(tmp_path / "customer_timeline.sqlite", tmp_path)
    config = service_config_from_json(write_service_config(tmp_path, enabled=False))

    report = run_nightly_service(config)

    assert report["overall_status"] == "partial"
    assert report["steps"][0]["status"] == "failed_required_disabled"
    assert report["failed_required_steps"] == ["local_jsonl"]
    assert report["snapshot_manifest"]["latest_published"] is False
    assert report["snapshot_manifest"]["counts"]["timeline_events"] == 0


def test_nightly_service_rejects_enabled_unknown_step(tmp_path: Path) -> None:
    seed_customer(tmp_path / "customer_timeline.sqlite", tmp_path)
    config_path = write_service_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["steps"][0]["kind"] = "shell"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported enabled step kind"):
        service_config_from_json(config_path)


def test_launchd_install_scripts_are_dry_run_by_default(tmp_path: Path) -> None:
    plist = tmp_path / "service.plist"
    plist.write_text("<plist/>", encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]

    install = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "install_customer_timeline_nightly_service.sh"),
            "--plist",
            str(plist),
            "--target",
            str(tmp_path / "target.plist"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    uninstall = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "uninstall_customer_timeline_nightly_service.sh"),
            "--target",
            str(tmp_path / "target.plist"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert install.returncode == 0
    assert uninstall.returncode == 0
    assert "Dry-run only" in install.stdout
    assert "Dry-run only" in uninstall.stdout
    assert not (tmp_path / "target.plist").exists()


def test_nightly_service_cli_summary_only(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    config_path = write_service_config(tmp_path)
    repo = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_customer_timeline_nightly_service.py"),
            "--config",
            str(config_path),
            "--summary-only",
        ],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(repo / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["steps"][0]["summary"]["changed_customer_count"] == 1
    assert payload["snapshot_manifest"]["counts"]["timeline_events"] == 1
