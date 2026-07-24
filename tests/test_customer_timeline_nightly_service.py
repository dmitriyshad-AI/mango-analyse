from __future__ import annotations

from contextlib import contextmanager
import json
import os
import plistlib
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import mango_mvp.customer_timeline.nightly_service as nightly_service_module
from mango_mvp.customer_timeline import (
    CustomerIdentity,
    CustomerTimelineSQLiteStore,
    IdentityLink,
    IdentityLinkType,
    IdentityMatchClass,
    IdentityStatus,
)
from mango_mvp.customer_timeline.nightly_service import (
    NightlyServiceStep,
    run_nightly_service,
    run_tallanto_money_api_step,
    service_config_from_json,
)


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


def seed_phone_link(db_path: Path, allowed_root: Path, *, phone: str = "+79990001122") -> None:
    with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                link_type=IdentityLinkType.PHONE,
                link_value=phone,
                source_system="test",
                source_ref="test:phone",
                customer_id="customer:nightly-1",
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                first_seen_at=NOW,
                last_seen_at=NOW,
            )
        )


def write_processed_call_db(
    path: Path,
    *,
    phone: str = "+79990001122",
    rows: tuple[dict[str, object], ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE call_records (
              id TEXT PRIMARY KEY,
              source_call_id TEXT,
              source_filename TEXT,
              started_at TEXT,
              phone TEXT,
              manager_name TEXT,
              direction TEXT,
              duration_sec REAL,
              analysis_status TEXT,
              analysis_json TEXT
            )
            """
        )
        rows = rows or (
            {
                "id": "call-row-1",
                "source_call_id": "provider-call-1",
                "started_at": "2026-07-04T10:00:00+00:00",
                "analysis_status": "done",
            },
        )
        for row in rows:
            con.execute(
                """
                INSERT INTO call_records (
                  id, source_call_id, source_filename, started_at, phone, manager_name,
                  direction, duration_sec, analysis_status, analysis_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["source_call_id"],
                    f"{row['id']}.wav",
                    row["started_at"],
                    phone,
                    "manager",
                    "inbound",
                    120.0,
                    row["analysis_status"],
                    json.dumps(
                        {
                            "summary": "Клиент спросил про расписание.",
                            "history_summary": "Клиент спросил про расписание.",
                            "call_type": "sales_call",
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        con.commit()


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


def test_nightly_service_runs_optional_amo_incremental_without_copying_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {
            "cursor_before": {"amo_leads_updated_at": "2026-07-01T00:00:00+00:00"},
            "cursor_after": {"amo_leads_updated_at": "2026-07-02T00:00:00+00:00"},
            "fetch": {"amo_leads_updated_at": {"page_cap_hit": False}},
            "repeat_run_duplicates": 0,
            "safety": {"amo_write": False, "tallanto_write": False, "crm_write": False},
            "first_run": {"cards": {"source_errors": []}, "events": {"source_errors": []}},
            "second_run": {"source_errors": []},
        }

    monkeypatch.setattr(nightly_service_module, "run_amo_incremental", fake_run)
    config_path = tmp_path / "amo_service_config.json"
    config_path.write_text(
        json.dumps(
            {
                "timeline_db": str(db_path),
                "allowed_root": str(tmp_path),
                "out_root": str(tmp_path / "nightly_service"),
                "publish_dir": str(tmp_path / "published"),
                "tenant_id": "foton",
                "steps": [
                    {
                        "name": "amo_incremental_shadow",
                        "kind": "amo_incremental",
                        "enabled": True,
                        "required": False,
                        "config": {
                            "out_root": str(tmp_path / "amo_incremental"),
                            "mcp_env": str(tmp_path / "amo.env"),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["overall_status"] == "ok"
    assert report["steps"][0]["status"] == "ok"
    assert report["steps"][0]["summary"]["repeat_run_duplicates"] == 0
    assert captured["config"].timeline_db == db_path
    assert captured["config"].copy_db is False


def test_nightly_service_runs_required_tallanto_attendance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = tmp_path / ".codex_local" / "staging"
    staging.mkdir(parents=True)
    db_path = staging / "customer_timeline.sqlite"
    seed_customer(db_path, staging)
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {"validation_ok": True, "counts": {"resolved": 2}, "safety": {"network_calls": False}}

    monkeypatch.setattr(nightly_service_module, "run_tallanto_attendance_import", fake_run)
    config_path = staging / "service.json"
    config_path.write_text(
        json.dumps(
            {
                "timeline_db": str(db_path),
                "allowed_root": str(staging),
                "out_root": str(staging / "runs"),
                "publish_dir": str(staging / "published"),
                "steps": [
                    {
                        "name": "tallanto_attendance",
                        "kind": "tallanto_attendance",
                        "required": True,
                        "config": {
                            "contacts_workbook": str(tmp_path / "contacts.xlsx"),
                            "attendance_report": str(tmp_path / "attendance.xlsx"),
                            "apply": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["overall_status"] == "ok"
    assert report["steps"][0]["summary"]["counts"]["resolved"] == 2
    assert captured["config"].timeline_db == db_path


def test_nightly_service_runs_wappi_then_refreshes_family_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / ".codex_local" / "staging"
    staging.mkdir(parents=True)
    db_path = staging / "customer_timeline.sqlite"
    seed_customer(db_path, staging)
    calls: list[str] = []

    def fake_wappi(config):
        calls.append("wappi")
        assert config.require_widget_linkage is True
        assert config.limits.show_all_chats is True
        return {"validation_ok": True, "summary": {"records_built": 3}}

    def fake_family(config):
        calls.append("family")
        assert config.apply is True
        return {"quick_check": "ok", "family_members_write_applied": True}

    monkeypatch.setattr(nightly_service_module, "run_wappi_history_import", fake_wappi)
    monkeypatch.setattr(nightly_service_module, "build_family_graph", fake_family)
    config_path = staging / "service.json"
    config_path.write_text(
        json.dumps(
            {
                "timeline_db": str(db_path),
                "allowed_root": str(staging),
                "out_root": str(staging / "runs"),
                "publish_dir": str(staging / "published"),
                "steps": [
                    {
                        "name": "wappi_history_incremental",
                        "kind": "wappi_history",
                        "config": {
                            "env_file": str(tmp_path / "wappi.env"),
                            "phase1_config": str(tmp_path / "phase1.json"),
                            "widget_link_db": str(staging / "wappi_links.sqlite"),
                            "require_widget_linkage": True,
                            "show_all_chats": True,
                        },
                    },
                    {
                        "name": "family_graph_refresh",
                        "kind": "family_graph",
                        "config": {
                            "out_path": str(staging / "family_graph.json"),
                            "apply": True,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["overall_status"] == "ok"
    assert calls == ["wappi", "family"]
    assert report["steps"][0]["summary"]["records_built"] == 3
    assert report["steps"][1]["summary"]["family_members_write_applied"] is True


def test_nightly_service_amo_incremental_failure_is_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    monkeypatch.setattr(
        nightly_service_module,
        "run_amo_incremental",
        lambda _config: (_ for _ in ()).throw(TimeoutError("AMO unavailable")),
    )
    config_path = tmp_path / "amo_service_config.json"
    config_path.write_text(
        json.dumps(
            {
                "timeline_db": str(db_path),
                "allowed_root": str(tmp_path),
                "out_root": str(tmp_path / "nightly_service"),
                "publish_dir": str(tmp_path / "published"),
                "steps": [
                    {
                        "name": "amo_incremental_shadow",
                        "kind": "amo_incremental",
                        "enabled": True,
                        "required": False,
                        "config": {
                            "out_root": str(tmp_path / "amo_incremental"),
                            "mcp_env": str(tmp_path / "amo.env"),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["overall_status"] == "ok"
    assert report["steps"][0]["status"] == "skipped_optional_failed"
    assert report["steps"][0]["error_type"] == "TimeoutError"
    assert report["snapshot_manifest"]["latest_published"] is True


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


def test_nightly_service_runs_tallanto_money_api_importer_without_exposing_env(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    importer = tmp_path / "repo/scripts/import_tallanto_payments_to_timeline.py"
    importer.parent.mkdir(parents=True)
    importer.write_text("# test importer\n", encoding="utf-8")
    env_file = tmp_path / "tallanto.env"
    env_file.write_text("CRM_TALLANTO_API_TOKEN=secret\n", encoding="utf-8")
    timeline_db = tmp_path / "customer_timeline.sqlite"
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "validation_ok": True,
                    "summary": {"status": "completed", "records_loaded": 2},
                    "api": {"modules": {}},
                    "safety": {
                        "write_tallanto": False,
                        "write_product_timeline_db": True,
                    },
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(nightly_service_module.subprocess, "run", fake_run)
    step = NightlyServiceStep(
        name="tallanto_money_api_incremental",
        kind="tallanto_money_api",
        tallanto_money_api_config={
            "importer_script": str(importer),
            "tallanto_env_file": str(env_file),
            "timeline_db": str(timeline_db),
            "allowed_root": str(tmp_path),
            "apply": True,
        },
    )

    report = run_tallanto_money_api_step(
        step,
        timeline_db=timeline_db,
        allowed_root=tmp_path,
        tenant_id="foton",
    )

    assert report["validation_ok"] is True
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--tallanto-api-env") + 1] == str(env_file)
    assert "secret" not in " ".join(command)


def test_nightly_service_sweeps_processed_mango_call_dbs_before_import(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    seed_phone_link(db_path, tmp_path)
    source_root = tmp_path / "product_data"
    call_db = source_root / "mango_update_after_20260704_20260704_v1" / "asr_ui_batch" / "calls.sqlite"
    write_processed_call_db(call_db)
    out_jsonl = tmp_path / "nightly_dv2_sources" / "mango_processed_sweep.jsonl"
    config_payload = {
        "timeline_db": str(db_path),
        "allowed_root": str(tmp_path),
        "out_root": str(tmp_path / "nightly_service"),
        "publish_dir": str(tmp_path / "published"),
        "tenant_id": "foton",
        "steps": [
            {
                "name": "mango_processed_sweep",
                "kind": "mango_processed_sweep",
                "enabled": True,
                "config": {
                    "producer_script": str(Path(__file__).resolve().parents[1] / "scripts" / "build_mango_call_timeline_increment.py"),
                    "scan_roots": [str(source_root)],
                    "package_globs": ["mango_update_after_*"],
                    "out_jsonl": str(out_jsonl),
                    "report_out": str(tmp_path / "nightly_dv2_sources" / "mango_processed_sweep_report.json"),
                    "manifest_path": str(tmp_path / "nightly_dv2_sources" / "mango_processed_sweep_manifest.json"),
                    "inventory_out": str(tmp_path / "nightly_dv2_sources" / "mango_processed_sweep_inventory.json"),
                },
            },
            {
                "name": "calls_and_amo_incremental",
                "kind": "nightly_incremental",
                "enabled": True,
                "config": {
                    "journal_path": str(tmp_path / "nightly_service" / "journal.jsonl"),
                    "safety_margin_seconds": 0,
                    "sources": [
                        {
                            "name": "mango_processed_sweep",
                            "source_system": "mango_processed_summary",
                            "path": str(out_jsonl),
                            "source_ref": "mango:processed_sweep:latest",
                            "normalizer": "mango_processed_summary",
                        }
                    ],
                },
            },
        ],
    }
    config_path = tmp_path / "mango_sweep_service_config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    config = service_config_from_json(config_path)

    first = run_nightly_service(config)
    second = run_nightly_service(config)

    assert first["overall_status"] == "ok"
    assert first["steps"][0]["summary"]["events_written"] == 1
    assert first["steps"][1]["summary"]["changed_customer_count"] == 1
    assert second["overall_status"] == "ok"
    assert second["steps"][0]["summary"]["events_written"] == 1
    assert second["steps"][1]["summary"]["changed_customer_count"] == 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        row = con.execute(
            """
            SELECT COUNT(*)
            FROM timeline_events
            WHERE source_system = 'mango_processed_summary' AND event_type = 'mango_call'
            """
        ).fetchone()
        chunk_row = con.execute(
            """
            SELECT COUNT(*)
            FROM bot_context_chunks
            WHERE source_system = 'mango_processed_summary' AND allowed_for_bot = 0 AND requires_manager_review = 1
            """
        ).fetchone()
    assert row[0] == 1
    assert chunk_row[0] == 1


def test_nightly_service_sweeps_explicit_ready_package_db_before_import(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    seed_phone_link(db_path, tmp_path)
    ready_db = tmp_path / "drop" / "mango_calls_ready.sqlite"
    write_processed_call_db(ready_db)
    out_jsonl = tmp_path / "nightly_dv2_sources" / "mango_processed_sweep.jsonl"
    config_payload = {
        "timeline_db": str(db_path),
        "allowed_root": str(tmp_path),
        "out_root": str(tmp_path / "nightly_service"),
        "publish_dir": str(tmp_path / "published"),
        "tenant_id": "foton",
        "steps": [
            {
                "name": "mango_processed_sweep",
                "kind": "mango_processed_sweep",
                "enabled": True,
                "config": {
                    "producer_script": str(
                        Path(__file__).resolve().parents[1]
                        / "scripts"
                        / "build_mango_call_timeline_increment.py"
                    ),
                    "scan_roots": [],
                    "package_dbs": [str(ready_db)],
                    "out_jsonl": str(out_jsonl),
                    "report_out": str(tmp_path / "nightly_dv2_sources" / "producer_report.json"),
                    "manifest_path": str(tmp_path / "nightly_dv2_sources" / "manifest.json"),
                    "inventory_out": str(tmp_path / "nightly_dv2_sources" / "inventory.json"),
                },
            },
            {
                "name": "calls_and_amo_incremental",
                "kind": "nightly_incremental",
                "enabled": True,
                "config": {
                    "journal_path": str(tmp_path / "nightly_service" / "journal.jsonl"),
                    "safety_margin_seconds": 0,
                    "sources": [
                        {
                            "name": "mango_processed_sweep",
                            "source_system": "mango_processed_summary",
                            "path": str(out_jsonl),
                            "source_ref": "mango:processed_sweep:latest",
                            "normalizer": "mango_processed_summary",
                        }
                    ],
                },
            },
        ],
    }
    config_path = tmp_path / "explicit_ready_service_config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["overall_status"] == "ok"
    assert report["steps"][0]["summary"]["events_written"] == 1
    assert report["steps"][1]["summary"]["changed_customer_count"] == 1
    inventory = json.loads(
        (tmp_path / "nightly_dv2_sources" / "inventory.json").read_text(encoding="utf-8")
    )
    assert [item["db_path"] for item in inventory] == [str(ready_db.resolve())]


def test_nightly_service_imports_late_analyzed_old_call_once(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    seed_phone_link(db_path, tmp_path)
    ready_db = tmp_path / "drop" / "mango_calls_ready.sqlite"
    write_processed_call_db(
        ready_db,
        rows=(
            {
                "id": "new-done",
                "source_call_id": "provider-new-done",
                "started_at": "2026-07-04T10:00:00+00:00",
                "analysis_status": "done",
            },
            {
                "id": "old-pending",
                "source_call_id": "provider-old-pending",
                "started_at": "2026-06-01T10:00:00+00:00",
                "analysis_status": "pending",
            },
        ),
    )
    out_jsonl = tmp_path / "nightly_dv2_sources" / "mango_processed_sweep.jsonl"
    config_payload = {
        "timeline_db": str(db_path),
        "allowed_root": str(tmp_path),
        "out_root": str(tmp_path / "nightly_service"),
        "publish_dir": str(tmp_path / "published"),
        "tenant_id": "foton",
        "steps": [
            {
                "name": "mango_processed_sweep",
                "kind": "mango_processed_sweep",
                "enabled": True,
                "config": {
                    "producer_script": str(
                        Path(__file__).resolve().parents[1]
                        / "scripts"
                        / "build_mango_call_timeline_increment.py"
                    ),
                    "package_dbs": [str(ready_db)],
                    "out_jsonl": str(out_jsonl),
                    "report_out": str(tmp_path / "nightly_dv2_sources/producer_report.json"),
                    "manifest_path": str(tmp_path / "nightly_dv2_sources/manifest.json"),
                    "inventory_out": str(tmp_path / "nightly_dv2_sources/inventory.json"),
                },
            },
            {
                "name": "calls_and_amo_incremental",
                "kind": "nightly_incremental",
                "enabled": True,
                "config": {
                    "journal_path": str(tmp_path / "nightly_service/journal.jsonl"),
                    "sources": [
                        {
                            "name": "mango_processed_sweep",
                            "source_system": "mango_processed_summary",
                            "path": str(out_jsonl),
                            "source_ref": "mango:processed_sweep:latest",
                            "normalizer": "mango_processed_summary",
                            "ignore_cursor": True,
                            "preserve_cursor": True,
                        }
                    ],
                },
            },
        ],
    }
    config_path = tmp_path / "late_call_service_config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    config = service_config_from_json(config_path)

    first = run_nightly_service(config)
    with sqlite3.connect(ready_db) as con:
        con.execute(
            "UPDATE call_records SET analysis_status = 'done' WHERE id = 'old-pending'"
        )
        con.commit()
    second = run_nightly_service(config)
    third = run_nightly_service(config)

    assert first["steps"][1]["summary"]["changed_customer_count"] == 1
    assert second["steps"][1]["summary"]["changed_customer_count"] == 1
    assert third["steps"][1]["summary"]["changed_customer_count"] == 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        count = con.execute(
            """
            SELECT COUNT(*)
            FROM timeline_events
            WHERE source_system = 'mango_processed_summary' AND event_type = 'mango_call'
            """
        ).fetchone()[0]
    assert count == 2


def test_nightly_service_fails_when_explicit_ready_package_db_is_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    out_jsonl = tmp_path / "nightly_dv2_sources" / "mango_processed_sweep.jsonl"
    config_payload = {
        "timeline_db": str(db_path),
        "allowed_root": str(tmp_path),
        "out_root": str(tmp_path / "nightly_service"),
        "publish_dir": str(tmp_path / "published"),
        "tenant_id": "foton",
        "steps": [
            {
                "name": "mango_processed_sweep",
                "kind": "mango_processed_sweep",
                "enabled": True,
                "required": True,
                "config": {
                    "producer_script": str(
                        Path(__file__).resolve().parents[1]
                        / "scripts"
                        / "build_mango_call_timeline_increment.py"
                    ),
                    "package_dbs": [str(tmp_path / "missing/mango_calls_ready.sqlite")],
                    "out_jsonl": str(out_jsonl),
                    "report_out": str(tmp_path / "nightly_dv2_sources/producer_report.json"),
                    "manifest_path": str(tmp_path / "nightly_dv2_sources/manifest.json"),
                    "inventory_out": str(tmp_path / "nightly_dv2_sources/inventory.json"),
                },
            }
        ],
    }
    config_path = tmp_path / "missing_ready_service_config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["overall_status"] == "partial"
    assert report["failed_required_steps"] == ["mango_processed_sweep"]
    assert report["steps"][0]["status"] == "failed"


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


def test_nightly_service_runs_required_tallanto_attendance_api_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / ".codex_local" / "staging" / "customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    seed_customer(db_path, tmp_path)
    config_path = tmp_path / "tallanto_api_service.json"
    config_path.write_text(
        json.dumps(
            {
                "timeline_db": str(db_path),
                "allowed_root": str(tmp_path),
                "out_root": str(tmp_path / "runs"),
                "publish_dir": str(tmp_path / "published"),
                "steps": [{
                    "name": "tallanto_attendance_api_incremental",
                    "kind": "tallanto_attendance_api",
                    "required": True,
                    "config": {
                        "tallanto_env_file": "~/.mango_secrets/tallanto_readonly.env",
                        "initial_since": "2026-07-13T00:00:00+03:00",
                        "apply": True,
                    },
                }],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        nightly_service_module,
        "run_tallanto_attendance_api_increment",
        lambda config: {
            "validation_ok": True,
            "cursor_before": config.initial_since.isoformat(),
            "cursor_after": "2026-07-24T10:00:00+03:00",
            "counts": {"created": 1},
            "safety": {"writes_tallanto": False},
        },
    )

    config = service_config_from_json(config_path)
    report = run_nightly_service(config)

    assert config.steps[0].tallanto_attendance_api_config is not None
    assert config.steps[0].tallanto_attendance_api_config.tallanto_env_file == Path(
        "~/.mango_secrets/tallanto_readonly.env"
    ).expanduser()
    assert report["overall_status"] == "ok"
    assert report["steps"][0]["status"] == "ok"
    assert report["steps"][0]["summary"]["counts"] == {"created": 1}


def test_nightly_service_marks_required_tallanto_partial_and_does_not_publish_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / ".codex_local" / "staging" / "customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    seed_customer(db_path, tmp_path)
    config_path = tmp_path / "tallanto_api_service.json"
    config_path.write_text(
        json.dumps(
            {
                "timeline_db": str(db_path),
                "allowed_root": str(tmp_path),
                "out_root": str(tmp_path / "runs"),
                "publish_dir": str(tmp_path / "published"),
                "steps": [
                    {
                        "name": "tallanto_attendance_api_incremental",
                        "kind": "tallanto_attendance_api",
                        "required": True,
                        "config": {
                            "tallanto_env_file": "~/.mango_secrets/tallanto_readonly.env",
                            "initial_since": "2026-07-13T00:00:00+03:00",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        nightly_service_module,
        "run_tallanto_attendance_api_increment",
        lambda config: {
            "status": "partial",
            "validation_ok": False,
            "unresolved_count": 2,
            "cursor_before": config.initial_since.isoformat(),
            "cursor_after": config.initial_since.isoformat(),
            "counts": {"created": 1, "identity_unmatched": 2},
            "safety": {"writes_tallanto": False},
        },
    )

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["overall_status"] == "partial"
    assert report["steps"][0]["status"] == "partial"
    assert report["steps"][0]["summary"]["unresolved_count"] == 2
    assert report["failed_required_steps"] == ["tallanto_attendance_api_incremental"]
    assert report["snapshot_manifest"]["latest_published"] is False


def test_nightly_service_runs_terminal_bot_safe_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / ".codex_local" / "staging" / "customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    seed_customer(db_path, tmp_path)
    config_path = tmp_path / "bot_safe_rebuild_service.json"
    config_path.write_text(
        json.dumps(
            {
                "timeline_db": str(db_path),
                "allowed_root": str(tmp_path),
                "out_root": str(tmp_path / "runs"),
                "publish_dir": str(tmp_path / "published"),
                "steps": [
                    {
                        "name": "bot_safe_rebuild",
                        "kind": "bot_safe_rebuild",
                        "required": True,
                        "config": {"apply": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class Report:
        def to_json_dict(self):
            return {
                "considered_customers": 3,
                "customers_with_summary": 2,
                "created": 1,
                "updated": 1,
                "duplicate": 0,
                "retired_stale": 1,
            }

    monkeypatch.setattr(nightly_service_module, "build_bot_safe_summaries", lambda config: Report())

    config = service_config_from_json(config_path)
    report = run_nightly_service(config)

    assert config.steps[0].bot_safe_rebuild_config is not None
    assert config.steps[0].bot_safe_rebuild_config.apply is True
    assert report["overall_status"] == "ok"
    assert report["steps"][0]["status"] == "ok"
    assert report["steps"][0]["summary"]["customers_with_summary"] == 2


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


def test_launchd_installer_renders_code_root_before_bootstrap(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "template.plist"
    target = tmp_path / "target.plist"
    code_root = tmp_path / "permanent&main"
    source.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
<key>Label</key><string>test.render</string>
<key>WorkingDirectory</key><string>__MANGO_CODE_ROOT__</string>
<key>EnvironmentVariables</key><dict>
<key>CUSTOMER_TIMELINE_NIGHTLY_HOME</key><string>__CUSTOMER_TIMELINE_NIGHTLY_HOME__</string>
</dict>
</dict></plist>
""",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_launchctl.chmod(0o700)

    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts/install_customer_timeline_nightly_service.sh"),
            "--plist",
            str(source),
            "--target",
            str(target),
            "--code-root",
            str(code_root),
            "--nightly-home",
            str(tmp_path / "persistent-nightly"),
            "--apply",
        ],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "__MANGO_CODE_ROOT__" not in target.read_text(encoding="utf-8")
    rendered = plistlib.loads(target.read_bytes())
    assert rendered["WorkingDirectory"] == str(code_root.resolve())
    assert rendered["EnvironmentVariables"]["CUSTOMER_TIMELINE_NIGHTLY_HOME"] == str(
        (tmp_path / "persistent-nightly").resolve()
    )


def test_launchd_templates_use_persistent_runtime_and_no_old_worktree() -> None:
    repo = Path(__file__).resolve().parents[1]
    paths = [
        repo / "deploy/customer_timeline_nightly/com.mango.customer-timeline-nightly.plist.template",
        repo / "deploy/customer_timeline_daily_captures/com.mango.customer-timeline-mail-chain.plist.template",
        repo / "deploy/customer_timeline_daily_captures/com.mango.customer-timeline-mango-capture.plist.template",
        repo / "deploy/customer_timeline_daily_captures/com.mango.customer-timeline-tallanto-api-capture.plist.template",
    ]
    for path in paths:
        payload = plistlib.loads(path.read_bytes())
        text = path.read_text(encoding="utf-8")
        assert "Mango_email_pipeline_restore" not in text
        assert payload["EnvironmentVariables"]["CUSTOMER_TIMELINE_NIGHTLY_HOME"] == (
            "__CUSTOMER_TIMELINE_NIGHTLY_HOME__"
        )
        assert payload["WorkingDirectory"] == "__MANGO_CODE_ROOT__"
    tallanto = plistlib.loads(paths[-1].read_bytes())
    assert tallanto["EnvironmentVariables"]["TALLANTO_API_CAPTURE_ENABLED"] == "1"


def test_deprecated_mail_capture_template_cannot_be_scheduled() -> None:
    repo = Path(__file__).resolve().parents[1]
    path = repo / "deploy/customer_timeline_daily_captures/com.mango.customer-timeline-mail-capture.plist.template"
    payload = plistlib.loads(path.read_bytes())

    assert payload["Disabled"] is True
    assert "StartCalendarInterval" not in payload
    assert payload["ProgramArguments"] == ["/usr/bin/false"]


def test_customer_timeline_deploy_templates_do_not_reference_retired_worktree() -> None:
    repo = Path(__file__).resolve().parents[1]
    roots = [
        repo / "deploy/customer_timeline_daily_captures",
        repo / "deploy/customer_timeline_nightly",
    ]
    for root in roots:
        for path in root.glob("*.plist.template"):
            assert "Mango_email_pipeline_restore" not in path.read_text(encoding="utf-8")


def test_daily_capture_launchd_templates_and_drivers_are_safe_by_default(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    templates = [
        repo / "deploy" / "customer_timeline_daily_captures" / "com.mango.customer-timeline-mail-capture.plist.template",
        repo / "deploy" / "customer_timeline_daily_captures" / "com.mango.customer-timeline-mango-capture.plist.template",
        repo / "deploy" / "customer_timeline_nightly" / "com.mango.customer-timeline-nightly.plist.template",
    ]
    for template in templates:
        lint = subprocess.run(["plutil", "-lint", str(template)], text=True, capture_output=True, check=False)
        assert lint.returncode == 0, lint.stderr + lint.stdout

    mail_manifest = tmp_path / "mail_manifest.json"
    mango_manifest = tmp_path / "mango_manifest.json"
    mail = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "run_customer_timeline_mail_capture_daily.sh"),
            "--lock-dir",
            str(tmp_path / "mail.lock"),
            "--manifest",
            str(mail_manifest),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    mango = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "run_customer_timeline_mango_capture_daily.sh"),
            "--lock-dir",
            str(tmp_path / "mango.lock"),
            "--manifest",
            str(mango_manifest),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert mail.returncode == 0
    assert mango.returncode == 0
    assert json.loads(mail_manifest.read_text(encoding="utf-8"))["status"] == "dry_run"
    mango_payload = json.loads(mango_manifest.read_text(encoding="utf-8"))
    assert mango_payload["status"] == "dry_run"
    assert mango_payload["runs_asr"] is False


def test_daily_capture_driver_blocks_double_run_with_lock(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    lock_dir = tmp_path / "mail.lock"
    first_manifest = tmp_path / "first.json"
    second_manifest = tmp_path / "second.json"
    first = subprocess.Popen(
        [
            "bash",
            str(repo / "scripts" / "run_customer_timeline_mail_capture_daily.sh"),
            "--lock-dir",
            str(lock_dir),
            "--manifest",
            str(first_manifest),
            "--hold-lock-seconds",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        import time

        deadline = time.time() + 1.5
        while time.time() < deadline and not lock_dir.exists():
            time.sleep(0.05)
        second = subprocess.run(
            [
                "bash",
                str(repo / "scripts" / "run_customer_timeline_mail_capture_daily.sh"),
                "--lock-dir",
                str(lock_dir),
                "--manifest",
                str(second_manifest),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert second.returncode == 75
        assert json.loads(second_manifest.read_text(encoding="utf-8"))["status"] == "locked"
    finally:
        first.communicate(timeout=5)


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
