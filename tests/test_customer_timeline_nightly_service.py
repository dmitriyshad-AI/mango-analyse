from __future__ import annotations

from contextlib import contextmanager
import json
import os
import plistlib
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
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
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.nightly_service import (
    NightlyServiceStep,
    _SourceProofContext,
    _proof_family_child_graph,
    run_nightly_service,
    run_tallanto_money_api_step,
    service_config_from_json,
)


NOW = datetime(2026, 7, 3, 3, 20, tzinfo=timezone.utc)


def family_graph_proof(summary: dict[str, object]) -> dict[str, object]:
    return dict(
        _proof_family_child_graph(
            _SourceProofContext(
                steps_by_name={"family_graph_refresh": {"status": "ok", "summary": summary}},
                source_counts=(),
                cursors=(),
                mail_link_enrich={},
                now=NOW,
            )
        )
    )


def test_family_graph_proof_rejects_zero_child_links() -> None:
    proof = family_graph_proof(
        {
            "quick_check": "ok",
            "family_links_total": 0,
            "customers_with_family_links": 0,
            "family_members_total": 34533,
        }
    )

    assert proof["status"] == "empty"
    assert proof["records_seen_or_written"] == 0


def test_family_graph_proof_uses_current_or_preserved_link_count() -> None:
    current = family_graph_proof({"quick_check": "ok", "family_links_total": 12})
    preserved = family_graph_proof(
        {
            "quick_check": "ok",
            "family_links_total": 0,
            "existing_family_links": 9,
            "child_graph_preserved_without_profiles": True,
        }
    )

    assert current["status"] == "ok"
    assert current["records_seen_or_written"] == 12
    assert preserved["status"] == "ok"
    assert preserved["records_seen_or_written"] == 9


def test_repo_python_env_removes_parent_git_context(monkeypatch, tmp_path) -> None:
    for key in nightly_service_module.GIT_CONTEXT_ENV_KEYS:
        monkeypatch.setenv(key, f"hostile-{key.lower()}")

    env = nightly_service_module._repo_python_env(tmp_path)

    assert all(key not in env for key in nightly_service_module.GIT_CONTEXT_ENV_KEYS)
    assert str(tmp_path / "src") in env["PYTHONPATH"].split(os.pathsep)


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
            "validation_ok": True,
            "complete": True,
            "cursor_before": {"amo_leads_updated_at": "2026-07-01T00:00:00+00:00"},
            "cursor_after": {"amo_leads_updated_at": "2026-07-02T00:00:00+00:00"},
            "fetch": {
                key: {"page_cap_hit": False, "complete": True, "pagination_drift_detected": False}
                for key in ("amo_leads_updated_at", "amo_contacts_updated_at", "amo_events_created_at")
            },
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


def test_amo_incremental_report_ok_rejects_partial_or_drifting_fetch() -> None:
    base = {
        "validation_ok": True,
        "complete": True,
        "safety": {"amo_write": False, "tallanto_write": False, "crm_write": False},
        "fetch": {
            key: {"complete": True, "page_cap_hit": False, "pagination_drift_detected": False}
            for key in ("amo_leads_updated_at", "amo_contacts_updated_at", "amo_events_created_at")
        },
        "first_run": {"cards": {"source_errors": []}, "events": {"source_errors": []}},
        "second_run": {"source_errors": []},
    }
    assert nightly_service_module.amo_incremental_report_ok(base) is True
    assert nightly_service_module.amo_incremental_report_ok({**base, "validation_ok": False}) is False
    drift = json.loads(json.dumps(base))
    drift["fetch"]["amo_events_created_at"]["pagination_drift_detected"] = True
    assert nightly_service_module.amo_incremental_report_ok(drift) is False


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


def test_nightly_service_does_not_publish_when_wappi_identity_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / ".codex_local" / "staging"
    staging.mkdir(parents=True)
    db_path = staging / "customer_timeline.sqlite"
    seed_customer(db_path, staging)
    monkeypatch.setattr(
        nightly_service_module,
        "run_wappi_history_import",
        lambda _config: {
            "validation_ok": True,
            "fetch_complete": True,
            "attribution_complete": False,
            "publish_ready": False,
            "summary": {"messages_newly_saved": 1, "pending_attribution": 1},
        },
    )
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
                        "required": True,
                        "config": {
                            "env_file": str(tmp_path / "wappi.env"),
                            "phase1_config": str(tmp_path / "phase1.json"),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["steps"][0]["status"] == "failed"
    assert report["overall_status"] == "partial"
    assert report["snapshot_manifest"]["latest_published"] is False


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


def test_tallanto_money_failure_diagnostic_contains_no_raw_output(tmp_path: Path, monkeypatch) -> None:
    importer = tmp_path / "repo/scripts/import_tallanto_payments_to_timeline.py"
    importer.parent.mkdir(parents=True)
    importer.write_text("# test importer\n", encoding="utf-8")
    env_file = tmp_path / "tallanto.env"
    env_file.write_text("CRM_TALLANTO_API_TOKEN=secret\n", encoding="utf-8")
    timeline_db = tmp_path / "staging.sqlite"
    raw_stdout = "parent@example.com +7 916 111-22-33"
    raw_stderr = "authorization=super-secret-token"
    monkeypatch.setattr(
        nightly_service_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 9, raw_stdout, raw_stderr),
    )
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

    with pytest.raises(RuntimeError) as raised:
        run_tallanto_money_api_step(
            step,
            timeline_db=timeline_db,
            allowed_root=tmp_path,
            tenant_id="foton",
        )

    diagnostics_path = Path(raised.value.diagnostics_path)
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload["returncode"] == 9
    assert payload["raw_output_persisted"] is False
    assert raw_stdout not in diagnostics_path.read_text(encoding="utf-8")
    assert raw_stderr not in diagnostics_path.read_text(encoding="utf-8")
    assert diagnostics_path.stat().st_mode & 0o777 == 0o600


def test_nightly_service_blocks_tallanto_money_when_cards_step_is_not_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / ".codex_local/staging/customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    seed_customer(db_path, tmp_path)
    env_file = tmp_path / "tallanto.env"
    env_file.write_text("CRM_TALLANTO_API_TOKEN=secret\n", encoding="utf-8")
    importer = tmp_path / "repo/scripts/import_tallanto_payments_to_timeline.py"
    importer.parent.mkdir(parents=True)
    importer.write_text("# not called\n", encoding="utf-8")
    money_called = False

    monkeypatch.setattr(
        nightly_service_module,
        "run_tallanto_cards_sync",
        lambda config: {"validation_ok": False, "apply_blocked": True, "blocked_reason": "synthetic"},
    )

    def fail_if_money_runs(*args, **kwargs):
        nonlocal money_called
        money_called = True
        raise AssertionError("money must not run after failed cards")

    monkeypatch.setattr(nightly_service_module, "run_tallanto_money_api_step", fail_if_money_runs)
    config_path = tmp_path / "nightly.json"
    config_path.write_text(
        json.dumps(
            {
                "timeline_db": str(db_path),
                "allowed_root": str(tmp_path),
                "out_root": str(tmp_path / "runs"),
                "publish_dir": str(tmp_path / "published"),
                "steps": [
                    {
                        "name": "tallanto_cards_sync",
                        "kind": "tallanto_cards",
                        "required": True,
                        "config": {
                            "timeline_db": str(db_path),
                            "allowed_root": str(tmp_path),
                            "out_root": str(tmp_path / "cards"),
                            "tallanto_env_file": str(env_file),
                        },
                    },
                    {
                        "name": "tallanto_money_api_incremental",
                        "kind": "tallanto_money_api",
                        "required": True,
                        "config": {
                            "importer_script": str(importer),
                            "tallanto_env_file": str(env_file),
                            "timeline_db": str(db_path),
                            "allowed_root": str(tmp_path),
                            "apply": True,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_nightly_service(service_config_from_json(config_path))

    assert money_called is False
    assert report["steps"][1]["status"] == "failed"
    assert report["steps"][1]["reason"] == "upstream_not_ok:tallanto_cards_sync"


def test_nightly_service_refreshes_existing_purchase_view_after_tallanto_money(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / ".codex_local/staging/customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    seed_customer(db_path, tmp_path)
    env_file = tmp_path / "tallanto.env"
    env_file.write_text("CRM_TALLANTO_API_TOKEN=secret\n", encoding="utf-8")
    importer = tmp_path / "repo/scripts/import_tallanto_payments_to_timeline.py"
    importer.parent.mkdir(parents=True)
    importer.write_text("# mocked\n", encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        nightly_service_module,
        "run_tallanto_money_api_step",
        lambda *args, **kwargs: {
            "validation_ok": True,
            "summary": {"status": "completed", "records_loaded": 1},
            "safety": {"write_tallanto": False, "write_product_timeline_db": True},
        },
    )

    def fake_refresh(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return {"rows_upserted": 1, "money_kind": {"fact": 1}}

    monkeypatch.setattr(nightly_service_module, "refresh_customer_purchases_v1", fake_refresh)
    config_path = tmp_path / "nightly.json"
    config_path.write_text(
        json.dumps(
            {
                "timeline_db": str(db_path),
                "allowed_root": str(tmp_path),
                "out_root": str(tmp_path / "runs"),
                "publish_dir": str(tmp_path / "published"),
                "steps": [
                    {
                        "name": "tallanto_money_api_incremental",
                        "kind": "tallanto_money_api",
                        "required": True,
                        "config": {
                            "importer_script": str(importer),
                            "tallanto_env_file": str(env_file),
                            "timeline_db": str(db_path),
                            "allowed_root": str(tmp_path),
                            "apply": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_nightly_service(service_config_from_json(config_path))

    assert captured["path"] == db_path
    assert report["steps"][0]["status"] == "ok"
    assert report["steps"][0]["summary"]["customer_purchases_v1"]["rows_upserted"] == 1


@pytest.mark.parametrize(
    ("stdout", "failure_kind"),
    (
        ("not-json", "invalid_json"),
        ("[]", "non_object_report"),
        (json.dumps({"safety": {"write_tallanto": True}}), "safety_contract_failed"),
    ),
)
def test_tallanto_money_contract_failures_create_safe_diagnostics(
    tmp_path: Path, monkeypatch, stdout: str, failure_kind: str
) -> None:
    importer = tmp_path / "repo/scripts/import_tallanto_payments_to_timeline.py"
    importer.parent.mkdir(parents=True)
    importer.write_text("# test importer\n", encoding="utf-8")
    env_file = tmp_path / "tallanto.env"
    env_file.write_text("token=secret\n", encoding="utf-8")
    timeline_db = tmp_path / "staging.sqlite"
    monkeypatch.setattr(
        nightly_service_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout, "parent@example.com"),
    )
    step = NightlyServiceStep(
        name="tallanto_money_api_incremental",
        kind="tallanto_money_api",
        tallanto_money_api_config={
            "importer_script": str(importer), "tallanto_env_file": str(env_file),
            "timeline_db": str(timeline_db), "allowed_root": str(tmp_path), "apply": True,
        },
    )

    with pytest.raises(RuntimeError) as raised:
        run_tallanto_money_api_step(step, timeline_db=timeline_db, allowed_root=tmp_path, tenant_id="foton")

    payload = json.loads(Path(raised.value.diagnostics_path).read_text(encoding="utf-8"))
    assert payload["failure_kind"] == failure_kind
    assert payload["raw_output_persisted"] is False


def test_tallanto_money_timeout_creates_safe_diagnostics(tmp_path: Path, monkeypatch) -> None:
    importer = tmp_path / "repo/scripts/import_tallanto_payments_to_timeline.py"
    importer.parent.mkdir(parents=True)
    importer.write_text("# test importer\n", encoding="utf-8")
    env_file = tmp_path / "tallanto.env"
    env_file.write_text("token=secret\n", encoding="utf-8")
    timeline_db = tmp_path / "staging.sqlite"

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 5, output="+79161112233", stderr="secret@example.com")

    monkeypatch.setattr(nightly_service_module.subprocess, "run", timeout)
    step = NightlyServiceStep(
        name="tallanto_money_api_incremental",
        kind="tallanto_money_api",
        tallanto_money_api_config={
            "importer_script": str(importer), "tallanto_env_file": str(env_file),
            "timeline_db": str(timeline_db), "allowed_root": str(tmp_path), "apply": True,
        },
    )

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run_tallanto_money_api_step(step, timeline_db=timeline_db, allowed_root=tmp_path, tenant_id="foton")

    payload = json.loads(Path(raised.value.diagnostics_path).read_text(encoding="utf-8"))
    assert payload["failure_kind"] == "timeout"
    assert payload["raw_output_persisted"] is False


def test_required_tallanto_money_proof_rejects_success_without_events() -> None:
    result = nightly_service_module.check_required_manifest_sources(
        [{"name": "tallanto_money_api_incremental", "status": "ok"}],
        ["tallanto_payments_subscriptions"],
        source_counts=[],
        now=NOW,
    )

    proof = result["proofs"]["tallanto_payments_subscriptions"]
    assert proof["status"] == "missing"
    assert result["missing"] == ["tallanto_payments_subscriptions"]


def test_nightly_service_reports_tallanto_money_diagnostics_path(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / ".codex_local/staging/customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    seed_customer(db_path, tmp_path)
    importer = tmp_path / "repo/scripts/import_tallanto_payments_to_timeline.py"
    importer.parent.mkdir(parents=True)
    importer.write_text("# test importer\n", encoding="utf-8")
    env_file = tmp_path / "tallanto.env"
    env_file.write_text("CRM_TALLANTO_API_TOKEN=secret\n", encoding="utf-8")
    monkeypatch.setattr(
        nightly_service_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 9, "raw phone +79161112233", "raw token"),
    )
    config_path = tmp_path / "nightly.json"
    config_path.write_text(
        json.dumps(
            {
                "timeline_db": str(db_path),
                "allowed_root": str(tmp_path),
                "out_root": str(tmp_path / "runs"),
                "publish_dir": str(tmp_path / "published"),
                "steps": [
                    {
                        "name": "tallanto_money_api_incremental",
                        "kind": "tallanto_money_api",
                        "required": True,
                        "config": {
                            "importer_script": str(importer),
                            "tallanto_env_file": str(env_file),
                            "timeline_db": str(db_path),
                            "allowed_root": str(tmp_path),
                            "apply": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_nightly_service(service_config_from_json(config_path))

    step_report = report["steps"][0]
    assert step_report["status"] == "failed"
    assert Path(step_report["error_diagnostics_path"]).is_file()


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


def _service_report_path_on_disk(report: dict) -> Path:
    return Path(report["out_root"]) / f"run_{report['run_id']}" / "service_report.json"


def test_nightly_service_local_freshness_monitor_exception_is_reported_and_next_step_still_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    source_path = tmp_path / "source.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "source_id": "nightly-event-after-monitor-crash",
                "customer_id": "customer:nightly-1",
                "event_type": "system_note",
                "event_at": "2026-07-03T03:00:00+00:00",
                "updated_at": "2026-07-03T03:00:00+00:00",
                "direction": "system",
                "summary": "Идёт после упавшего монитора.",
            }
        ],
    )
    monkeypatch.setattr(
        nightly_service_module,
        "run_local_freshness_monitor",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom token=SECRET_ABC123")),
    )
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
                "config": {"paths": []},
            },
            {
                "name": "local_jsonl",
                "kind": "nightly_incremental",
                "enabled": True,
                "required": True,
                "config": {
                    "journal_path": str(tmp_path / "nightly_service" / "journal.jsonl"),
                    "safety_margin_seconds": 60,
                    "sources": [
                        {
                            "name": "local_jsonl",
                            "source_system": "nightly_test_source",
                            "path": str(source_path),
                            "source_ref": "test:nightly-after-crash",
                            "normalizer": "jsonl",
                        }
                    ],
                },
            },
        ],
    }
    config_path = tmp_path / "monitor_exception_optional_config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["steps"][0]["status"] == "skipped_optional_failed"
    assert report["steps"][0]["error_type"] == "RuntimeError"
    assert report["steps"][0]["reason"] == "step_exception:RuntimeError"
    assert "boom" not in json.dumps(report)
    assert "SECRET_ABC123" not in json.dumps(report)
    # the next, independent stage still ran despite the optional stage's exception.
    assert report["steps"][1]["status"] == "ok"
    assert report["steps"][1]["summary"]["changed_customer_count"] == 1
    assert report["overall_status"] == "ok"
    assert report["failed_required_steps"] == []
    assert report["snapshot_manifest"]["latest_published"] is True
    # the service report is always written to disk, even though a stage raised.
    service_report_path = _service_report_path_on_disk(report)
    assert service_report_path.exists()
    on_disk_text = service_report_path.read_text(encoding="utf-8")
    assert json.loads(on_disk_text)["steps"][0]["error_type"] == "RuntimeError"
    assert "boom" not in on_disk_text
    assert "SECRET_ABC123" not in on_disk_text


def test_nightly_service_local_freshness_monitor_exception_blocks_latest_publish_when_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    monkeypatch.setattr(
        nightly_service_module,
        "run_local_freshness_monitor",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
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
                "required": True,
                "config": {"paths": []},
            }
        ],
    }
    config_path = tmp_path / "monitor_exception_required_config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["overall_status"] == "partial"
    assert report["steps"][0]["status"] == "failed"
    assert report["steps"][0]["error_type"] == "RuntimeError"
    assert report["failed_required_steps"] == ["wappi_history_incremental"]
    assert report["snapshot_manifest"]["latest_published"] is False
    assert not (tmp_path / "published" / "latest_customer_timeline_snapshot.json").exists()
    # even a required-stage exception must not skip writing the service report.
    assert _service_report_path_on_disk(report).exists()


def test_nightly_service_mango_processed_sweep_exception_is_reported_and_next_step_still_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    monkeypatch.setattr(
        nightly_service_module,
        "run_mango_processed_sweep",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom secret=XYZ789")),
    )
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
                "required": False,
                "config": {},
            },
            {
                "name": "freshness_after_sweep_crash",
                "kind": "local_freshness_monitor",
                "enabled": True,
                "required": True,
                "config": {"paths": [], "empty_status": "ok"},
            },
        ],
    }
    config_path = tmp_path / "sweep_exception_optional_config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["steps"][0]["status"] == "skipped_optional_failed"
    assert report["steps"][0]["error_type"] == "RuntimeError"
    assert "boom" not in json.dumps(report)
    assert "secret=XYZ789" not in json.dumps(report)
    # the next, independent stage still ran despite the optional stage's exception.
    assert report["steps"][1]["status"] == "ok"
    assert report["overall_status"] == "ok"
    assert report["snapshot_manifest"]["latest_published"] is True
    assert _service_report_path_on_disk(report).exists()


def test_nightly_service_mango_processed_sweep_exception_blocks_latest_publish_when_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    monkeypatch.setattr(
        nightly_service_module,
        "run_mango_processed_sweep",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
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
                "config": {},
            }
        ],
    }
    config_path = tmp_path / "sweep_exception_required_config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["overall_status"] == "partial"
    assert report["steps"][0]["status"] == "failed"
    assert report["steps"][0]["error_type"] == "RuntimeError"
    assert report["failed_required_steps"] == ["mango_processed_sweep"]
    assert report["snapshot_manifest"]["latest_published"] is False
    assert _service_report_path_on_disk(report).exists()


# --- BLOCK B: bounded timeout, resume, idempotent repeat, fail-loud manifest ---


def test_nightly_service_step_timeout_stops_hanging_external_script(tmp_path: Path) -> None:
    """B2 proof 1: a step's external subprocess cannot hang the run forever."""
    importer = tmp_path / "repo" / "scripts" / "import_tallanto_payments_to_timeline.py"
    importer.parent.mkdir(parents=True)
    importer.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    env_file = tmp_path / "tallanto.env"
    env_file.write_text("CRM_TALLANTO_API_TOKEN=secret\n", encoding="utf-8")
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    config_path = tmp_path / "timeout_service_config.json"
    config_path.write_text(
        json.dumps(
            {
                "timeline_db": str(db_path),
                "allowed_root": str(tmp_path),
                "out_root": str(tmp_path / "nightly_service"),
                "publish_dir": str(tmp_path / "published"),
                "step_timeout_seconds": 0.5,
                "steps": [
                    {
                        "name": "tallanto_money_api_incremental",
                        "kind": "tallanto_money_api",
                        "required": True,
                        "config": {
                            "importer_script": str(importer),
                            "tallanto_env_file": str(env_file),
                            "timeline_db": str(db_path),
                            "allowed_root": str(tmp_path),
                            "apply": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    started = time.monotonic()
    report = run_nightly_service(service_config_from_json(config_path))
    elapsed = time.monotonic() - started

    assert elapsed < 4.0, "step must not block for the full 5s external sleep"
    assert report["steps"][0]["status"] == "failed"
    assert "TimeoutExpired" in report["steps"][0]["reason"]
    assert report["overall_status"] == "partial"
    assert report["snapshot_manifest"]["latest_published"] is False


def test_nightly_service_resumes_from_last_completed_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B2 proof 2: an interrupted run is resumed, not restarted from step 1."""
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    out_root = tmp_path / "nightly_service"
    source_path = tmp_path / "source.jsonl"
    write_jsonl(
        source_path,
        [
            {
                "source_id": "resume-event-1",
                "customer_id": "customer:nightly-1",
                "event_type": "system_note",
                "event_at": "2026-07-03T03:00:00+00:00",
                "updated_at": "2026-07-03T03:00:00+00:00",
                "direction": "system",
                "summary": "Событие после resume.",
            }
        ],
    )
    config_payload = {
        "timeline_db": str(db_path),
        "allowed_root": str(tmp_path),
        "out_root": str(out_root),
        "publish_dir": str(tmp_path / "published"),
        "steps": [
            {
                "name": "amo_incremental_shadow",
                "kind": "amo_incremental",
                "required": True,
                "config": {
                    "out_root": str(tmp_path / "amo_incremental"),
                    "mcp_env": str(tmp_path / "amo.env"),
                },
            },
            {
                "name": "local_jsonl",
                "kind": "nightly_incremental",
                "required": True,
                "config": {
                    "journal_path": str(out_root / "journal.jsonl"),
                    "safety_margin_seconds": 60,
                    "sources": [
                        {
                            "name": "local_jsonl",
                            "source_system": "nightly_test_source",
                            "path": str(source_path),
                            "source_ref": "test:nightly-resume",
                            "normalizer": "jsonl",
                        }
                    ],
                },
            },
        ],
    }
    config_path = tmp_path / "resume_service_config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    config = service_config_from_json(config_path)
    resolved_db, resolved_root, resolved_out, resolved_publish = nightly_service_module.validated_service_paths(config)
    config_fingerprint = nightly_service_module.service_config_fingerprint(
        config,
        timeline_db=resolved_db,
        allowed_root=resolved_root,
        out_root=resolved_out,
        publish_dir=resolved_publish,
    )

    # Seed a crashed prior run: step 1 completed ok, step 2 never started.
    prior_run_id = "20260101T000000Z"
    prior_run_dir = out_root / f"run_{prior_run_id}"
    prior_run_dir.mkdir(parents=True)
    prior_step_report = {
        "index": 1,
        "name": "amo_incremental_shadow",
        "kind": "amo_incremental",
        "status": "ok",
        "required": True,
        "report_path": str(prior_run_dir / "01_amo_incremental_shadow.json"),
        "summary": {"repeat_run_duplicates": 0},
        "duration_seconds": 1.2,
    }
    (prior_run_dir / "progress.json").write_text(
        json.dumps(
            {
                "schema_version": "customer_timeline_nightly_service_progress_v2",
                "run_id": prior_run_id,
                "total_steps": 2,
                "completed_steps": 1,
                "config_fingerprint": config_fingerprint,
                "steps": [prior_step_report],
                # B2: matches the DB's real on-disk state right now (nothing
                # has touched db_path since seed_customer() above), proving
                # resume is accepted when the checkpoint genuinely matches.
                "db_checkpoint": nightly_service_module.db_lightweight_checkpoint(resolved_db),
            }
        ),
        encoding="utf-8",
    )

    def fail_if_called(amo_config):  # pragma: no cover - proves resume skips completed steps
        raise AssertionError("amo_incremental_shadow must be skipped on resume")

    monkeypatch.setattr(nightly_service_module, "run_amo_incremental", fail_if_called)

    report = run_nightly_service(config)

    assert report["run_id"] == prior_run_id
    assert report["resumed_from_run_id"] == prior_run_id
    assert report["steps"][0] == prior_step_report
    assert report["steps"][1]["name"] == "local_jsonl"
    assert report["steps"][1]["status"] == "ok"
    assert report["overall_status"] == "ok"
    assert (prior_run_dir / "service_report.json").exists()


def test_nightly_service_second_run_over_fixtures_has_no_duplicate_events(tmp_path: Path) -> None:
    """B2 proof 3: a second, fully-completed run over the same fixtures is a
    clean independent run (not a resume) and writes zero duplicate rows."""
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    config = service_config_from_json(write_service_config(tmp_path))

    first = run_nightly_service(config)
    second = run_nightly_service(config)

    assert first["overall_status"] == "ok"
    assert second["overall_status"] == "ok"
    assert second["resumed_from_run_id"] is None
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        count = con.execute(
            "SELECT COUNT(*) FROM timeline_events WHERE source_system = 'nightly_test_source'"
        ).fetchone()[0]
    assert count == 1


def test_nightly_service_fails_loud_when_required_manifest_source_is_missing(tmp_path: Path) -> None:
    """B2 proof 4: no false PASS when a mandatory business source is missing.

    Before this fix, a config that never ran the Wappi step still reported
    overall_status "ok" and published "latest" -- the exact silent-success
    behaviour the launchd job was observed exhibiting.
    """
    # family_graph with apply=True requires the DB under .codex_local/staging
    # (see family_graph._guard_db); mirrors the other real (non-mocked)
    # family_graph test in this file.
    staging = tmp_path / ".codex_local" / "staging"
    staging.mkdir(parents=True)
    db_path = staging / "customer_timeline.sqlite"
    seed_customer(db_path, staging)
    config_path = tmp_path / "required_sources_config.json"
    config_payload = {
        "timeline_db": str(db_path),
        "allowed_root": str(staging),
        "out_root": str(staging / "nightly_service"),
        "publish_dir": str(staging / "published"),
        "required_manifest_sources": ["family_child_graph", "wappi_telegram"],
        "steps": [
            {
                "name": "family_graph_refresh",
                "kind": "family_graph",
                "required": True,
                "config": {
                    "timeline_db": str(db_path),
                    "allowed_root": str(staging),
                    "apply": True,
                },
            }
        ],
    }
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = run_nightly_service(service_config_from_json(config_path))

    assert report["steps"][0]["name"] == "family_graph_refresh"
    assert report["steps"][0]["status"] == "ok"
    assert report["overall_status"] == "partial"
    assert "required_manifest_source:wappi_telegram" in report["failed_required_steps"]
    assert "required_manifest_source:family_child_graph" in report["failed_required_steps"]
    assert report["required_sources_check"]["missing"] == ["family_child_graph", "wappi_telegram"]
    assert report["required_sources_check"]["satisfied"] == []
    assert report["snapshot_manifest"]["latest_published"] is False


def test_nightly_service_wappi_proof_is_independent_per_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4 proof 1: wappi_history_incremental=ok must not vouch for both
    Telegram and MAX -- each channel needs its own evidence in
    timeline_events, so a channel with zero rows is reported missing even
    though the shared step succeeded (the exact false-green this fixes)."""
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    telegram_source = tmp_path / "wappi_telegram.jsonl"
    write_jsonl(
        telegram_source,
        [
            {
                "source_id": "tg-event-1",
                "customer_id": "customer:nightly-1",
                "event_type": "telegram_message",
                "event_at": "2026-07-03T03:00:00+00:00",
                "updated_at": "2026-07-03T03:00:00+00:00",
                "direction": "inbound",
                "summary": "Здравствуйте, расскажите про лагерь.",
            }
        ],
    )
    monkeypatch.setattr(
        nightly_service_module,
        "run_wappi_history_import",
        lambda config: {"validation_ok": True, "summary": {"records_built": 0}},
    )
    config_payload = {
        "timeline_db": str(db_path),
        "allowed_root": str(tmp_path),
        "out_root": str(tmp_path / "nightly_service"),
        "publish_dir": str(tmp_path / "published"),
        "required_manifest_sources": ["wappi_telegram", "wappi_max"],
        "steps": [
            {
                "name": "wappi_history_incremental",
                "kind": "wappi_history",
                "required": True,
                "config": {
                    "env_file": str(tmp_path / "wappi.env"),
                    "phase1_config": str(tmp_path / "phase1.json"),
                },
            },
            {
                "name": "telegram_seed",
                "kind": "nightly_incremental",
                "required": True,
                "config": {
                    "journal_path": str(tmp_path / "nightly_service" / "telegram_journal.jsonl"),
                    "safety_margin_seconds": 0,
                    "sources": [
                        {
                            "name": "telegram_seed",
                            "source_system": "wappi_telegram",
                            "path": str(telegram_source),
                            "source_ref": "test:wappi-telegram-seed",
                            "normalizer": "jsonl",
                        }
                    ],
                },
            },
        ],
    }
    config_path = tmp_path / "wappi_proof_config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = run_nightly_service(service_config_from_json(config_path))

    proofs = report["required_sources_check"]["proofs"]
    assert proofs["wappi_telegram"]["status"] == "ok"
    assert proofs["wappi_telegram"]["records_seen_or_written"] >= 1
    assert proofs["wappi_max"]["status"] == "missing"
    assert report["required_sources_check"]["missing"] == ["wappi_max"]
    assert report["overall_status"] == "partial"
    assert report["snapshot_manifest"]["latest_published"] is False


def _seed_tallanto_attendance_cursor(db_path: Path, *, updated_at: datetime) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT INTO ingestion_cursors (tenant_id, source_system, last_cursor_ts, updated_at, metadata_json)
            VALUES ('foton', 'tallanto_attendance_api', ?, ?, '{}')
            ON CONFLICT(tenant_id, source_system) DO UPDATE SET
              last_cursor_ts = excluded.last_cursor_ts, updated_at = excluded.updated_at
            """,
            (updated_at.isoformat(), updated_at.isoformat()),
        )
        con.commit()


def _seed_tallanto_attendance_event(db_path: Path, *, allowed_root: Path) -> None:
    now = datetime.now(timezone.utc)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:nightly-1",
                event_type=TimelineEventType.TALLANTO_ATTENDANCE,
                event_at=now,
                source_system="tallanto_attendance_api",
                source_id="existing-attendance",
                direction=TimelineDirection.SYSTEM,
                created_at=now,
            )
        )


def _tallanto_attendance_config(tmp_path: Path, db_path: Path) -> dict:
    return {
        "timeline_db": str(db_path),
        "allowed_root": str(tmp_path),
        "out_root": str(tmp_path / "runs"),
        "publish_dir": str(tmp_path / "published"),
        "required_manifest_sources": ["tallanto_attendance"],
        "steps": [
            {
                "name": "tallanto_attendance_api_incremental",
                "kind": "tallanto_attendance_api",
                "required": True,
                "config": {
                    "tallanto_env_file": "~/.mango_secrets/tallanto_readonly.env",
                    "initial_since": "2026-07-13T00:00:00+03:00",
                    "apply": True,
                },
            }
        ],
    }


def test_nightly_service_required_source_proof_passes_on_fresh_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4 proof 2: a source with zero new records this run but a freshly
    refreshed ingestion cursor still counts as proven-healthy -- the gate
    reads the real cursor (ground truth), not merely the step's own
    self-reported "ok"."""
    db_path = tmp_path / ".codex_local" / "staging" / "customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    seed_customer(db_path, tmp_path)
    _seed_tallanto_attendance_event(db_path, allowed_root=tmp_path)
    _seed_tallanto_attendance_cursor(db_path, updated_at=datetime.now(timezone.utc))
    config_path = tmp_path / "tallanto_api_service.json"
    config_path.write_text(
        json.dumps(_tallanto_attendance_config(tmp_path, db_path)), encoding="utf-8"
    )
    monkeypatch.setattr(
        nightly_service_module,
        "run_tallanto_attendance_api_increment",
        lambda config: {
            "validation_ok": True,
            "cursor_before": config.initial_since.isoformat(),
            "cursor_after": config.initial_since.isoformat(),
            "counts": {},
            "safety": {"writes_tallanto": False},
        },
    )

    report = run_nightly_service(service_config_from_json(config_path))

    proof = report["required_sources_check"]["proofs"]["tallanto_attendance"]
    assert proof["status"] == "ok", proof
    assert report["required_sources_check"]["missing"] == []
    assert report["overall_status"] == "ok"
    assert report["snapshot_manifest"]["latest_published"] is True


def test_nightly_service_required_attendance_proof_rejects_fresh_cursor_without_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / ".codex_local" / "staging" / "customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    seed_customer(db_path, tmp_path)
    _seed_tallanto_attendance_cursor(db_path, updated_at=datetime.now(timezone.utc))
    config_path = tmp_path / "tallanto_api_service.json"
    config_path.write_text(json.dumps(_tallanto_attendance_config(tmp_path, db_path)), encoding="utf-8")
    monkeypatch.setattr(
        nightly_service_module,
        "run_tallanto_attendance_api_increment",
        lambda config: {"validation_ok": True, "counts": {}, "safety": {"writes_tallanto": False}},
    )

    report = run_nightly_service(service_config_from_json(config_path))

    proof = report["required_sources_check"]["proofs"]["tallanto_attendance"]
    assert proof["status"] == "missing"
    assert report["snapshot_manifest"]["latest_published"] is False


def test_nightly_service_required_source_proof_fails_on_stale_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4 proof 3: the mirror image of the fresh case -- a step that
    self-reports "ok" every night but whose ingestion cursor has not
    actually moved in a long time must be caught as stale, not waved
    through on the step's say-so alone."""
    db_path = tmp_path / ".codex_local" / "staging" / "customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    seed_customer(db_path, tmp_path)
    _seed_tallanto_attendance_event(db_path, allowed_root=tmp_path)
    _seed_tallanto_attendance_cursor(
        db_path, updated_at=datetime.now(timezone.utc) - timedelta(hours=48)
    )
    config_path = tmp_path / "tallanto_api_service.json"
    config_path.write_text(
        json.dumps(_tallanto_attendance_config(tmp_path, db_path)), encoding="utf-8"
    )
    monkeypatch.setattr(
        nightly_service_module,
        "run_tallanto_attendance_api_increment",
        lambda config: {
            "validation_ok": True,
            "cursor_before": config.initial_since.isoformat(),
            "cursor_after": config.initial_since.isoformat(),
            "counts": {},
            "safety": {"writes_tallanto": False},
        },
    )

    report = run_nightly_service(service_config_from_json(config_path))

    proof = report["required_sources_check"]["proofs"]["tallanto_attendance"]
    assert proof["status"] == "stale", proof
    assert report["required_sources_check"]["missing"] == ["tallanto_attendance"]
    assert report["overall_status"] == "partial"
    assert report["snapshot_manifest"]["latest_published"] is False


def test_nightly_service_email_proof_fails_when_mail_archive_missing(tmp_path: Path) -> None:
    """B4 proof 4: the email source needs a genuine mail archive ingest, not
    just an ok mail_link_enrich step -- a missing archive input must fail
    the email proof."""
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    config_payload = {
        "timeline_db": str(db_path),
        "allowed_root": str(tmp_path),
        "out_root": str(tmp_path / "nightly_service"),
        "publish_dir": str(tmp_path / "published"),
        "required_manifest_sources": ["email"],
        "steps": [
            {
                "name": "mail_archive_incremental",
                "kind": "nightly_incremental",
                "required": True,
                "config": {
                    "journal_path": str(tmp_path / "nightly_service" / "mail_journal.jsonl"),
                    "safety_margin_seconds": 0,
                    "sources": [
                        {
                            "name": "mail_archive_stage2_incremental",
                            "source_system": "mail_archive_stage2",
                            "path": str(tmp_path / "missing_mail_archive.jsonl"),
                            "source_ref": "test:mail-archive-missing",
                            "normalizer": "mail_archive_stage2",
                            "required": True,
                        }
                    ],
                },
            },
            {
                "name": "mail_link_enrich",
                "kind": "mail_link_enrich",
                "required": True,
                "config": {},
            },
        ],
    }
    config_path = tmp_path / "email_proof_config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = run_nightly_service(service_config_from_json(config_path))

    proof = report["required_sources_check"]["proofs"]["email"]
    assert proof["status"] != "ok", proof
    assert report["required_sources_check"]["missing"] == ["email"]
    assert report["overall_status"] == "partial"
    assert report["snapshot_manifest"]["latest_published"] is False


def test_nightly_service_blocks_latest_when_quick_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B5 proof 1: a corrupted staging DB must never publish "latest", even
    when every individual step reports ok, and any previously published
    "latest" is left byte-for-byte untouched."""
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    config = service_config_from_json(write_service_config(tmp_path))
    latest_path = tmp_path / "published" / "latest_customer_timeline_snapshot.json"
    latest_path.parent.mkdir(parents=True)
    latest_path.write_text("OLD-LATEST-BYTES", encoding="utf-8")
    real_manifest = nightly_service_module.build_snapshot_manifest

    def corrupted_manifest(*args, **kwargs):
        manifest = dict(real_manifest(*args, **kwargs))
        manifest["quick_check"] = "corruption detected: *** in database main ***"
        return manifest

    monkeypatch.setattr(nightly_service_module, "build_snapshot_manifest", corrupted_manifest)

    report = run_nightly_service(config)

    assert report["overall_status"] == "partial"
    assert "timeline_db_quick_check" in report["failed_required_steps"]
    assert report["snapshot_manifest"]["latest_published"] is False
    assert latest_path.read_text(encoding="utf-8") == "OLD-LATEST-BYTES"


def test_nightly_service_latest_publish_survives_interruption_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B5 proof 2: if the process dies between writing the temp file and the
    final os.replace() for latest_customer_timeline_snapshot.json, the
    previous "latest" must be left exactly as it was -- never a truncated or
    half-written file at that path -- and no temp file litter remains."""
    db_path = tmp_path / "customer_timeline.sqlite"
    seed_customer(db_path, tmp_path)
    config = service_config_from_json(write_service_config(tmp_path))
    latest_path = tmp_path / "published" / "latest_customer_timeline_snapshot.json"
    latest_path.parent.mkdir(parents=True)
    latest_path.write_text("OLD-LATEST-BYTES", encoding="utf-8")

    real_replace = nightly_service_module.os.replace

    def flaky_replace(src, dst):
        if Path(dst) == latest_path:
            raise OSError("simulated interruption before replace")
        return real_replace(src, dst)

    monkeypatch.setattr(nightly_service_module.os, "replace", flaky_replace)

    with pytest.raises(OSError, match="simulated interruption"):
        run_nightly_service(config)

    assert latest_path.read_text(encoding="utf-8") == "OLD-LATEST-BYTES"
    leftover_tmp = [
        p for p in latest_path.parent.iterdir() if p.name.startswith(f".{latest_path.name}.")
    ]
    assert leftover_tmp == [], f"temp file(s) left behind: {leftover_tmp}"


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
