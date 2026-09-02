from __future__ import annotations

import importlib.util
import hashlib
import os
import sys
import json
import sqlite3
import textwrap
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pytest

from mango_mvp.productization.mail_archive import CANONICAL_MAIL_IDENTITY_DB
from mango_mvp.customer_timeline.nightly_service import (
    NIGHTLY_SERVICE_CONFIG_SCHEMA_VERSION,
    REQUIRED_MANIFEST_SOURCE_STEP_MAP,
    SOURCE_PROOF_BUILDERS,
)
from mango_mvp.customer_timeline.wappi_history_import import (
    WappiFetchLimits,
    profiles_from_phase1_config,
    wappi_fetch_universe_fingerprint,
    wappi_timeline_state,
)
from mango_mvp.integrations.amo_wappi_phase1 import AmoWappiPhase1Config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_customer_timeline_codex_task.py"
spec = importlib.util.spec_from_file_location("run_customer_timeline_codex_task", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)

BUILDER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_customer_timeline_nightly_dv2_sources.py"
builder_spec = importlib.util.spec_from_file_location("build_customer_timeline_nightly_dv2_sources_test", BUILDER_SCRIPT)
builder = importlib.util.module_from_spec(builder_spec)
assert builder_spec and builder_spec.loader
sys.modules[builder_spec.name] = builder
builder_spec.loader.exec_module(builder)
REAL_SNAPSHOT_WAPPI_PAIR_INPUTS = builder.snapshot_wappi_pair_inputs


@pytest.fixture(autouse=True)
def _configured_calls_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_root = tmp_path / "mango-calls-runtime"
    working_db = pipeline_root / "working" / "mango_calls_pipeline.sqlite"
    working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(working_db) as con:
        con.execute(
            "CREATE TABLE call_records (id TEXT PRIMARY KEY, analysis_status TEXT, analysis_json TEXT)"
        )
    service_config = tmp_path / "mango-calls-config.json"
    service_config.write_text(
        json.dumps({"pipeline_root": str(pipeline_root)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "DEFAULT_MANGO_CALLS_SERVICE_CONFIG", service_config)


@pytest.fixture(autouse=True)
def _stable_wrapper_wappi_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    phase1 = tmp_path / "wappi/phase1.json"
    phase1.parent.mkdir(parents=True, exist_ok=True)
    phase1.write_text('{"profiles": {}}\n', encoding="utf-8")
    monkeypatch.setattr(builder, "DEFAULT_WAPPI_CONFIG", phase1)

    def fake_snapshot(state_root: Path) -> dict:
        empty_sha = hashlib.sha256(b"[]\n").hexdigest()
        pairs_file, auto_pairs_file = builder.wappi_pair_snapshot_paths(
            state_root,
            pairs_sha256=empty_sha,
            auto_pairs_sha256=empty_sha,
        )
        pairs_file.parent.mkdir(parents=True, exist_ok=True)
        pairs_file.write_text("[]\n", encoding="utf-8")
        auto_pairs_file.write_text("[]\n", encoding="utf-8")
        manifest = pairs_file.parent / "pair_snapshot_manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return {
            "pairs_file": pairs_file,
            "auto_pairs_file": auto_pairs_file,
            "manifest": manifest,
            "files": {},
        }

    monkeypatch.setattr(module, "snapshot_wappi_pair_inputs", fake_snapshot)
    monkeypatch.setattr(builder, "snapshot_wappi_pair_inputs", fake_snapshot)


def test_lightweight_wrapper_matches_canonical_required_sources() -> None:
    canonical = set(REQUIRED_MANIFEST_SOURCE_STEP_MAP)

    assert set(module.REQUIRED_MANIFEST_SOURCES) == canonical
    assert set(SOURCE_PROOF_BUILDERS) == canonical
    assert module.EXPECTED_NIGHTLY_CONFIG_SCHEMA_VERSION == NIGHTLY_SERVICE_CONFIG_SCHEMA_VERSION


def _mail_root_with_identity(tmp_path: Path) -> Path:
    root = tmp_path / "mail-data"
    identity_db = root / CANONICAL_MAIL_IDENTITY_DB
    identity_db.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(identity_db).close()
    return root


def _write_verified_mail_process_state(state_root: Path) -> tuple[Path, Path, str]:
    mail_root = state_root / "mail_pipeline"
    process_root = mail_root / "process"
    process_root.mkdir(parents=True, exist_ok=True)
    output_jsonl = process_root / "mail_archive_stage2_incremental.jsonl"
    builder_manifest = process_root / "mail_increment_builder_manifest.json"
    download_manifest = mail_root / "mail_download_manifest.json"
    process_manifest = mail_root / "mail_process_manifest.json"
    output_jsonl.write_text("", encoding="utf-8")
    builder_manifest.write_text('{"status":"ok"}\n', encoding="utf-8")
    runtime = {"head": "test-head", "worktree": "/test/worktree"}
    download_manifest.write_text(
        json.dumps(
            {
                "status": "ok",
                "truncated": False,
                "errors": 0,
                "runtime": runtime,
                "mailbox_reports": {"inbox": {"status": "ok"}, "sent": {"status": "ok"}},
            }
        ),
        encoding="utf-8",
    )
    process_manifest.write_text(
        json.dumps(
            {
                "status": "ok",
                "finished_at": "2026-09-02T00:00:00+00:00",
                "rows_written": 0,
                "max_event_at": None,
                "output_jsonl": str(output_jsonl),
                "output_sha256": builder.sha256_file(output_jsonl),
                "builder_manifest": str(builder_manifest),
                "builder_manifest_sha256": builder.sha256_file(builder_manifest),
                "download_manifest": str(download_manifest),
                "download_manifest_sha256": builder.sha256_file(download_manifest),
                "runtime": runtime,
            }
        ),
        encoding="utf-8",
    )
    return output_jsonl, process_manifest, builder.sha256_file(process_manifest)


def test_builder_passes_mail_data_root_separately_from_repo_root(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    mail_data_root = tmp_path / "Mango_Data"
    identity_one = tmp_path / "readonly/tallanto_identity_1.sqlite"
    identity_two = tmp_path / "readonly/tallanto_identity_2.sqlite"
    identity_one.parent.mkdir(parents=True)
    identity_one.touch()
    identity_two.touch()
    captured = {}
    assert builder.build_parser().parse_args([]).mail_data_root == str(builder.DEFAULT_MAIL_DATA_ROOT)

    monkeypatch.setattr(builder, "build_mango_freshness", lambda *args, **kwargs: {})
    _write_verified_mail_process_state(tmp_path / "state")

    def fake_service_config(**kwargs):
        captured["mail_data_root"] = kwargs["mail_data_root"]
        captured["tallanto_identity_dbs"] = kwargs["tallanto_identity_dbs"]
        return {}

    monkeypatch.setattr(builder, "build_service_config", fake_service_config)

    result = builder.main(
        [
            "--source-root",
            str(repo_root),
            "--mail-data-root",
            str(mail_data_root),
            "--out-root",
            str(tmp_path / "out"),
            "--timeline-db",
            str(tmp_path / "timeline.sqlite"),
            "--tallanto-identity-db",
            str(identity_one),
            "--tallanto-identity-db",
            str(identity_two),
        ]
    )

    assert result == 0
    assert captured["mail_data_root"] == mail_data_root.resolve()
    assert captured["tallanto_identity_dbs"] == [str(identity_one), str(identity_two)]


def test_mail_builder_fails_before_writing_when_archive_input_is_missing(tmp_path) -> None:
    out_jsonl = tmp_path / "out.jsonl"
    manifest = tmp_path / "manifest.json"

    with pytest.raises(FileNotFoundError, match="required mail archive input is missing"):
        builder.build_mail_increment(
            tmp_path / "Mango_Data",
            out_jsonl=out_jsonl,
            manifest_path=manifest,
            since=builder.parse_dt("2026-06-19T14:53:27+00:00"),
            text_limit=1200,
        )

    assert not out_jsonl.exists()
    assert not manifest.exists()


def valid_nightly_payload(staging_root: Path) -> dict:
    timeline_db = staging_root / "customer_timeline_staging.sqlite"
    timeline_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(timeline_db) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS timeline_events ("
            "tenant_id TEXT NOT NULL, source_system TEXT NOT NULL, source_id TEXT NOT NULL)"
        )
    mail_identity_db = staging_root / "mail_identity.sqlite"
    mail_identity_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(mail_identity_db):
        pass
    empty_sha = hashlib.sha256(b"[]\n").hexdigest()
    pairs_file, auto_pairs_file = builder.wappi_pair_snapshot_paths(
        staging_root / "state",
        pairs_sha256=empty_sha,
        auto_pairs_sha256=empty_sha,
    )
    pairs_file.parent.mkdir(parents=True, exist_ok=True)
    pairs_file.write_text("[]\n", encoding="utf-8")
    auto_pairs_file.write_text("[]\n", encoding="utf-8")
    phase1_path = staging_root / "state/wappi/phase1.json"
    phase1_path.parent.mkdir(parents=True, exist_ok=True)
    phase1_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "p-tg": {"brand": "foton", "channel": "telegram", "label": "Foton"}
                }
            }
        ),
        encoding="utf-8",
    )
    mail_jsonl, mail_manifest, mail_manifest_sha = _write_verified_mail_process_state(
        staging_root / "state"
    )
    limits = WappiFetchLimits(
        chat_limit_per_profile=5000,
        messages_per_chat=50000,
        message_limit_total=50000,
        request_limit_total=50000,
        page_size=100,
        sleep_seconds=0.2,
        show_all_chats=True,
        complete_message_history=True,
    )
    profile = profiles_from_phase1_config(
        AmoWappiPhase1Config.from_file(phase1_path)
    )[0]
    timeline_state = wappi_timeline_state(
        timeline_db,
        tenant_id="foton",
        profiles=(profile,),
    )
    profile_key = f"{profile.source_system}:{profile.profile_id}"
    checkpoint_dir = staging_root / "state/wappi/checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "wappi_history_checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": "customer_timeline_wappi_history_checkpoint_v1",
                "profiles": {
                    profile_key: {
                        "fingerprint": wappi_fetch_universe_fingerprint(
                            profile, limits, tenant_id="foton"
                        ),
                        "complete": True,
                        "incremental_cycle": True,
                        "timeline_rows": 0,
                        "timeline_source_digest": timeline_state[profile_key]["source_digest"],
                        "catalog_next_offset": 0,
                        "chats_done": [],
                        "chat_markers": {},
                        "chat_cursors": {},
                        "full_audit_markers": {},
                        "active_chat": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "config_schema_version": module.EXPECTED_NIGHTLY_CONFIG_SCHEMA_VERSION,
        "required_manifest_sources": sorted(module.REQUIRED_MANIFEST_SOURCES),
        "timeline_db": str(staging_root / "customer_timeline_staging.sqlite"),
        "allowed_root": str(staging_root),
        "state_root": str(staging_root / "state"),
        "ownership_receipt_path": str(staging_root / "state/WRITER_OWNERSHIP.json"),
        "steps": [
            {
                "name": "mango_processed_sweep",
                "kind": "mango_processed_sweep",
                "enabled": True,
                "required": True,
                "config": {
                    "producer_script": str(module.ROOT / "scripts/build_mango_call_timeline_increment.py"),
                    "source_service_config": str(builder.DEFAULT_MANGO_CALLS_SERVICE_CONFIG),
                    "package_dbs": [
                        str(module.configured_calls_working_db(builder.DEFAULT_MANGO_CALLS_SERVICE_CONFIG))
                    ],
                },
            },
            {
                "name": "calls_and_amo_incremental",
                "kind": "nightly_incremental",
                "enabled": True,
                "required": True,
                "config": {
                    "journal_path": str(staging_root / "nightly_service/calls_and_amo.jsonl"),
                    "sources": [
                        {
                            "source_system": source_system,
                            "normalizer": normalizer,
                            "required": True,
                            "path": str(staging_root / "nightly_dv2_sources" / f"{source_system}.jsonl"),
                            **(
                                {"ignore_cursor": False, "preserve_cursor": False}
                                if source_system == "mango_processed_summary"
                                else {}
                            ),
                        }
                        for source_system, normalizer in module.REQUIRED_CALL_SOURCES.items()
                    ],
                },
            },
            {
                "name": "amo_incremental_shadow",
                "kind": "amo_incremental",
                "enabled": True,
                "required": True,
                "config": {"max_pages": 200, "page_limit": 20},
            },
            {
                "name": "wappi_history_incremental",
                "kind": "wappi_history",
                "enabled": True,
                "required": True,
                "config": {
                    "timeline_db": str(staging_root / "customer_timeline_staging.sqlite"),
                    "apply": True,
                    "require_widget_linkage": False,
                    "checkpoint_dir": str(checkpoint_dir),
                    "phase1_config": str(phase1_path),
                    "phase1_config_sha256": module.sha256_file(phase1_path),
                    "pairs_file": str(pairs_file),
                    "auto_pairs_file": str(auto_pairs_file),
                    "pairs_file_sha256": module.sha256_file(pairs_file),
                    "auto_pairs_file_sha256": module.sha256_file(auto_pairs_file),
                    "messages_per_chat": 50000,
                    "message_limit_total": 50000,
                    "request_limit_total": 50000,
                    "chat_limit_per_profile": 5000,
                    "page_size": 100,
                    "sleep_seconds": 0.2,
                    "show_all_chats": True,
                    "complete_message_history": True,
                },
            },
            {
                "name": "mail_archive_incremental",
                "kind": "nightly_incremental",
                "enabled": True,
                "required": True,
                "config": {
                    "sources": [
                        {
                            "source_system": "mail_archive_stage2",
                            "path": str(mail_jsonl),
                            "normalizer": "mail_archive_stage2",
                            "proof_manifest_path": str(mail_manifest),
                            "proof_manifest_sha256": mail_manifest_sha,
                            "proof_max_age_hours": 72.0,
                        }
                    ]
                },
            },
            {
                "name": "mail_link_enrich",
                "kind": "mail_link_enrich",
                "enabled": True,
                "required": True,
                "config": {
                    "reconsider_pending": True,
                    "tallanto_identity_dbs": [str(mail_identity_db)],
                },
            },
            {
                "name": "tallanto_money_api_incremental",
                "kind": "tallanto_money_api",
                "enabled": True,
                "required": True,
                "config": {
                    "importer_script": str(
                        module.ROOT / "scripts/import_tallanto_payments_to_timeline.py"
                    ),
                    "timeline_db": str(staging_root / "customer_timeline_staging.sqlite"),
                    "apply": True,
                    "timeout_seconds": 5400,
                },
            },
            {
                "name": "tallanto_cards_sync",
                "kind": "tallanto_cards",
                "enabled": True,
                "required": True,
                "config": {
                    "timeline_db": str(staging_root / "customer_timeline_staging.sqlite"),
                    "allowed_root": str(staging_root),
                    "out_root": str(staging_root / "tallanto_cards_sync"),
                    "tallanto_env_file": str(staging_root / "tallanto.env"),
                    "max_pages": module.DEFAULT_TALLANTO_CARDS_MAX_PAGES,
                },
            },
            {
                "name": "tallanto_attendance_api_incremental",
                "kind": "tallanto_attendance_api",
                "enabled": True,
                "required": True,
                "config": {
                    "timeline_db": str(staging_root / "customer_timeline_staging.sqlite"),
                    "apply": True,
                },
            },
            {
                "name": "family_graph_refresh",
                "kind": "family_graph",
                "enabled": True,
                "required": True,
                "config": {
                    "timeline_db": str(staging_root / "customer_timeline_staging.sqlite"),
                    "apply": True,
                },
            },
            {
                "name": "derived_signals_refresh",
                "kind": "derived_signals",
                "enabled": True,
                "required": True,
                "config": {
                    "timeline_db": str(staging_root / "customer_timeline_staging.sqlite"),
                    "apply": True,
                },
            },
            {
                "name": "stage4b_bot_opening",
                "kind": "stage4b_bot_opening",
                "enabled": True,
                "required": True,
                "config": {
                    "timeline_db": str(staging_root / "customer_timeline_staging.sqlite"),
                    "apply": True,
                },
            },
            {
                "name": "bot_safe_rebuild",
                "kind": "bot_safe_rebuild",
                "enabled": True,
                "required": True,
                "config": {
                    "timeline_db": str(staging_root / "customer_timeline_staging.sqlite"),
                    "apply": True,
                },
            },
        ],
    }
    steps = payload["steps"]
    tallanto_names = {
        "tallanto_cards_sync",
        "tallanto_money_api_incremental",
        "tallanto_attendance_api_incremental",
    }
    tallanto = {step["name"]: step for step in steps if step["name"] in tallanto_names}
    steps[:] = [step for step in steps if step["name"] not in tallanto_names]
    wappi_index = next(index for index, step in enumerate(steps) if step["name"] == "wappi_history_incremental")
    steps[wappi_index:wappi_index] = [
        tallanto["tallanto_cards_sync"],
        tallanto["tallanto_attendance_api_incremental"],
        tallanto["tallanto_money_api_incremental"],
    ]
    return payload


def config_step(payload: dict, name: str) -> dict:
    return next(step for step in payload["steps"] if step["name"] == name)


def move_config_step_before(payload: dict, name: str, before: str) -> None:
    steps = payload["steps"]
    moved = next(step for step in steps if step["name"] == name)
    steps.remove(moved)
    steps.insert(next(index for index, step in enumerate(steps) if step["name"] == before), moved)


def test_mail_existing_state_is_tenant_scoped(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE timeline_events (
              tenant_id TEXT,
              source_system TEXT,
              source_id TEXT,
              customer_id TEXT,
              match_status TEXT,
              confidence REAL,
              record_json TEXT
            )
            """
        )
        con.executemany(
            "INSERT INTO timeline_events VALUES (?, 'mail_archive_stage2', ?, ?, 'strong_unique', 1.0, '{}')",
            (("foton", "same", "customer-foton"), ("other", "same", "customer-other")),
        )

    state = builder.load_existing_mail_link_state(db_path, tenant_id="foton")
    source_ids = builder.load_existing_mail_source_ids(db_path, tenant_id="foton")

    assert state["same"]["customer_id"] == "customer-foton"
    assert source_ids == {"same"}


def test_tallanto_api_capture_is_fail_closed_without_explicit_env(monkeypatch) -> None:
    monkeypatch.delenv("TALLANTO_API_CAPTURE_ENABLED", raising=False)

    task = module.build_task_spec("tallanto-api-capture", tallanto_phone_limit=1)

    assert task.command == ()
    assert "not configured" in task.stop_reason


def test_mango_capture_uses_dry_run_when_command_file_missing(monkeypatch) -> None:
    monkeypatch.delenv("MANGO_CAPTURE_COMMAND_FILE", raising=False)

    task = module.build_task_spec("mango-capture", tallanto_phone_limit=1)

    assert task.command == ("bash", "scripts/run_customer_timeline_mango_capture_daily.sh")


def test_status_marks_partial_failure_as_stopped() -> None:
    status, reason = module.status_from_payload({"partial_failure": True}, 0, "")

    assert status == "stopped"
    assert reason == "partial_failure"


def test_status_marks_not_configured_as_stopped() -> None:
    status, reason = module.status_from_payload({"status": "not_configured"}, 0, "")

    assert status == "stopped"
    assert reason == "not_configured"


def test_status_marks_failed_incremental_gate_as_stopped() -> None:
    status, reason = module.status_from_payload(
        {"overall_status": "partial", "gate_passed": False, "failed_required_sources": ["mail"]},
        0,
        "",
    )

    assert status == "stopped"
    assert reason == "gate_failed"


def test_status_does_not_count_pass_with_notes_as_clean_cycle() -> None:
    status, reason = module.status_from_payload(
        {
            "overall_status": "ok",
            "partial_failure": False,
            "data_quality_status": "pass_with_notes",
        },
        0,
        "",
    )

    assert status == "stopped"
    assert reason == "data_quality_status=pass_with_notes"


def test_status_counts_explicit_data_quality_pass_as_clean_cycle() -> None:
    status, reason = module.status_from_payload(
        {
            "overall_status": "ok",
            "partial_failure": False,
            "data_quality_status": "pass",
        },
        0,
        "",
    )

    assert status == "ok"
    assert reason == ""


def test_status_reports_data_quality_before_command_rc() -> None:
    status, reason = module.status_from_payload(
        {
            "overall_status": "ok",
            "partial_failure": False,
            "data_quality_status": "pass_with_notes",
        },
        1,
        "",
        require_data_quality=True,
    )

    assert status == "stopped"
    assert reason == "data_quality_status=pass_with_notes"


def test_status_keeps_failed_command_when_data_quality_passes() -> None:
    status, reason = module.status_from_payload(
        {
            "overall_status": "ok",
            "partial_failure": False,
            "data_quality_status": "pass",
        },
        1,
        "",
        require_data_quality=True,
    )

    assert status == "failed"
    assert reason == "command_rc=1"


def test_nightly_wrapper_requires_explicit_data_quality_status() -> None:
    status, reason = module.status_from_payload(
        {"overall_status": "ok", "partial_failure": False},
        0,
        "",
        require_data_quality=True,
    )

    assert status == "stopped"
    assert reason == "data_quality_status_missing"


def test_nightly_wrapper_preserves_command_error_when_payload_is_missing() -> None:
    status, reason = module.status_from_payload(
        {},
        70,
        "",
        require_data_quality=True,
    )

    assert status == "failed"
    assert reason == "command_rc=70"


def test_mail_process_task_requires_fresh_download_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "MAIL_STATE_DIR", tmp_path)

    task = module.build_task_spec("mail-process", tallanto_phone_limit=1)

    assert task.command
    assert "stage manifest is missing" in task.stop_reason


def test_mail_import_is_not_available_as_a_scheduled_task() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        module.build_task_spec("mail-import", tallanto_phone_limit=1)


def test_mail_process_uses_persistent_staging_timeline_db(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "MAIL_STATE_DIR", tmp_path / "mail")
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", tmp_path / "customer_timeline_staging.sqlite")

    task = module.build_task_spec("mail-process", tallanto_phone_limit=1)

    assert task.command[-2:] == ("--timeline-db", str(tmp_path / "customer_timeline_staging.sqlite"))


def test_nightly_config_rejects_missing_required_calls_step(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    payload["steps"] = [step for step in payload["steps"] if step["name"] != "calls_and_amo_incremental"]
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    reason = module.validate_nightly_config(config)

    assert "calls_and_amo_incremental" in reason


def test_nightly_config_accepts_calls_step_without_optional_amo_sources(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    calls = next(step for step in payload["steps"] if step["name"] == "calls_and_amo_incremental")
    calls["config"]["sources"] = [
        source
        for source in calls["config"]["sources"]
        if source["source_system"] == "mango_processed_summary"
    ]
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    reason = module.validate_nightly_config(config)

    assert reason == ""


def test_nightly_config_rejects_changed_wappi_pair_snapshot(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    wappi = config_step(payload, "wappi_history_incremental")["config"]
    Path(wappi["auto_pairs_file"]).write_text('[{"changed": true}]\n', encoding="utf-8")
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert "snapshot SHA mismatch" in module.validate_nightly_config(config)


def test_nightly_config_rejects_oversized_amo_pages(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    amo = next(step for step in payload["steps"] if step["name"] == "amo_incremental_shadow")
    amo["config"]["page_limit"] = 50
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert "page_limit must be 20" in module.validate_nightly_config(config)


@pytest.mark.parametrize("step_name", [name for name, _kind in module.REQUIRED_MUTATING_NIGHTLY_CHAIN])
@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("enabled", "enabled and required"),
        ("required", "enabled and required"),
        ("kind", "kind must be"),
        ("timeline_db", "timeline_db must match"),
        ("apply", "apply must be true"),
    ],
)
def test_nightly_config_rejects_invalid_mutating_chain_step(
    tmp_path, monkeypatch, step_name: str, field: str, expected: str
) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    step = next(item for item in payload["steps"] if item["name"] == step_name)
    if field in {"enabled", "required"}:
        step[field] = False
    elif field == "kind":
        step["kind"] = "wrong_kind"
    elif field == "timeline_db":
        step["config"]["timeline_db"] = str(staging_root / "another.sqlite")
    else:
        step["config"]["apply"] = False
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert expected in module.validate_nightly_config(config)


@pytest.mark.parametrize(("mutation", "expected"), [("duplicate", "exactly one"), ("reorder", "step order")])
def test_nightly_config_rejects_duplicate_or_reordered_mutating_chain(
    tmp_path, monkeypatch, mutation: str, expected: str
) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    steps = payload["steps"]
    wappi_index = next(index for index, step in enumerate(steps) if step["name"] == "wappi_history_incremental")
    family_index = next(index for index, step in enumerate(steps) if step["name"] == "family_graph_refresh")
    if mutation == "duplicate":
        steps.insert(wappi_index + 1, dict(steps[wappi_index]))
    else:
        steps[wappi_index], steps[family_index] = steps[family_index], steps[wappi_index]
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert expected in module.validate_nightly_config(config)


@pytest.mark.parametrize(
    "step_name",
    [
        name
        for name, _kind in module.REQUIRED_CANONICAL_NIGHTLY_CHAIN
        if name not in {item[0] for item in module.REQUIRED_MUTATING_NIGHTLY_CHAIN}
    ],
)
def test_nightly_config_rejects_duplicate_core_source_owner(
    tmp_path, monkeypatch, step_name: str
) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    step = next(item for item in payload["steps"] if item["name"] == step_name)
    payload["steps"].append(dict(step))
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert f"exactly one {step_name}" in module.validate_nightly_config(config)


def test_nightly_config_rejects_renamed_second_amo_owner(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    amo = dict(next(item for item in payload["steps"] if item["name"] == "amo_incremental_shadow"))
    amo["name"] = "amo_incremental_second_owner"
    payload["steps"].append(amo)
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert "unexpected second owner for amo_incremental" in module.validate_nightly_config(config)


def test_nightly_config_rejects_tallanto_money_before_cards(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    steps = payload["steps"]
    cards_index = next(index for index, step in enumerate(steps) if step["name"] == "tallanto_cards_sync")
    money_index = next(index for index, step in enumerate(steps) if step["name"] == "tallanto_money_api_incremental")
    steps[cards_index], steps[money_index] = steps[money_index], steps[cards_index]
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert "cards -> attendance -> money" in module.validate_nightly_config(config)


def test_nightly_config_rejects_sweep_without_configured_calls_db(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    sweep = next(step for step in payload["steps"] if step["name"] == "mango_processed_sweep")
    sweep["config"]["package_dbs"] = []
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    reason = module.validate_nightly_config(config)

    assert "configured Mango Calls working DB" in reason


def test_nightly_config_resolves_calls_db_from_its_own_service_config(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    sweep = next(step for step in payload["steps"] if step["name"] == "mango_processed_sweep")
    override = tmp_path / "override-calls-config.json"
    override.write_text(
        json.dumps({"pipeline_root": str(tmp_path / "other-calls-runtime")}),
        encoding="utf-8",
    )
    sweep["config"]["source_service_config"] = str(override)
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert "only the configured Mango Calls working DB" in module.validate_nightly_config(config)


def test_nightly_config_rejects_missing_configured_calls_db(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    missing_root = tmp_path / "missing-mango-calls-runtime"
    builder.DEFAULT_MANGO_CALLS_SERVICE_CONFIG.write_text(
        json.dumps({"pipeline_root": str(missing_root)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    reason = module.validate_nightly_config(config)

    assert "is missing" in reason


def test_run_task_does_not_execute_command_after_preflight_stop(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(module, "TASK_STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(module, "FOTON_DAILY", tmp_path / "daily")
    monkeypatch.setattr(module, "ensure_nightly_config", lambda: "synthetic invalid config")
    monkeypatch.setattr(
        module,
        "build_task_spec",
        lambda *args, **kwargs: module.TaskSpec(
            task="nightly-warehouse",
            command=("python3", "must_not_run.py"),
            stop_reason=kwargs["nightly_stop_reason"],
        ),
    )
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append(args))

    rc = module.run_task("nightly-warehouse", tallanto_phone_limit=1)

    assert rc == 78
    assert calls == []


def test_nightly_wrapper_lock_covers_config_preflight(tmp_path, monkeypatch) -> None:
    held = False
    observations: list[bool] = []

    @contextmanager
    def fake_lock(path, *, timeout_seconds):
        nonlocal held
        assert path == tmp_path / "state/nightly-warehouse-wrapper"
        assert timeout_seconds == 0
        held = True
        try:
            yield {}
        finally:
            held = False

    def preflight() -> str:
        observations.append(held)
        return "synthetic stop"

    monkeypatch.setattr(module, "TASK_STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(module, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(module, "FOTON_DAILY", tmp_path / "daily")
    monkeypatch.setattr(module, "customer_timeline_run_lock", fake_lock)
    monkeypatch.setattr(module, "ensure_nightly_config", preflight)

    assert module.run_task("nightly-warehouse", tallanto_phone_limit=1) == 78
    assert observations == [True]


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_run_with_runtime_budget_kills_process_group_including_grandchild(tmp_path) -> None:
    """B3 proof: a hung nightly subprocess is bounded from the *outside* by
    killing its whole process group, not just its direct pid -- so a
    grandchild it shelled out to (e.g. a step's own external importer call)
    is reaped too, not left running as an orphan. Both the direct child and
    the grandchild install a SIGTERM-ignore handler, so only the SIGKILL
    escalation (not the initial SIGTERM) can end them; if the group kill
    were scoped to the direct pid only, the grandchild would still be alive
    after this call returns.
    """
    pid_file = tmp_path / "pids.txt"
    script = tmp_path / "hang_with_grandchild.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os, signal, subprocess, sys, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            with open({str(pid_file)!r}, "a") as fh:
                fh.write(str(os.getpid()) + "\\n")
            child = subprocess.Popen([
                sys.executable, "-c",
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
            ])
            with open({str(pid_file)!r}, "a") as fh:
                fh.write(str(child.pid) + "\\n")
            child.wait()
            """
        ),
        encoding="utf-8",
    )

    result = module.run_with_runtime_budget(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=os.environ,
        budget_seconds=0.3,
        term_grace_seconds=0.5,
    )

    assert result.timed_out is True
    pids = [int(line) for line in pid_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(pids) == 2, "both the direct child and the grandchild must have started"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(_pid_is_alive(pid) for pid in pids):
        time.sleep(0.1)
    alive = [pid for pid in pids if _pid_is_alive(pid)]
    assert alive == [], f"process(es) survived the process-group kill: {alive}"


def test_repo_python_env_removes_parent_git_context(monkeypatch) -> None:
    for key in module.GIT_CONTEXT_ENV_KEYS:
        monkeypatch.setenv(key, f"hostile-{key.lower()}")

    env = module.repo_python_env()

    assert all(key not in env for key in module.GIT_CONTEXT_ENV_KEYS)
    assert str(module.ROOT / "src") in env["PYTHONPATH"].split(os.pathsep)


def test_current_runtime_ignores_parent_git_context(monkeypatch, tmp_path) -> None:
    expected = module.current_runtime()
    for key in module.GIT_CONTEXT_ENV_KEYS:
        monkeypatch.setenv(key, str(tmp_path / f"hostile-{key.lower()}"))

    assert module.current_runtime() == expected


def test_nightly_task_timeout_reports_stopped_and_leaves_latest_untouched(tmp_path, monkeypatch) -> None:
    """B3 proof: run_task() enforces the runtime budget for nightly-warehouse
    specifically (not the unbounded subprocess.run other tasks still use),
    reports a clear timeout stop reason (not a bare "ok" or a misleading
    command_rc), and never touches the previous "latest" snapshot."""
    monkeypatch.setattr(module, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(module, "TASK_STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(module, "FOTON_DAILY", tmp_path / "daily")
    monkeypatch.setattr(module, "ensure_nightly_config", lambda: "")
    latest_path = tmp_path / "published" / "latest_customer_timeline_snapshot.json"
    latest_path.parent.mkdir(parents=True)
    latest_path.write_text("OLD-LATEST-BYTES", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "build_task_spec",
        lambda *args, **kwargs: module.TaskSpec(
            task="nightly-warehouse",
            command=("python3", "nightly.py"),
            expected_output=latest_path,
            stop_reason=kwargs["nightly_stop_reason"],
        ),
    )
    monkeypatch.setattr(module, "nightly_runtime_budget_seconds", lambda *a, **kw: 42.0)

    def fake_bounded_run(command, **kwargs):
        assert kwargs["budget_seconds"] == 42.0
        return module.BoundedRunResult(rc=-15, stdout="", timed_out=True)

    monkeypatch.setattr(module, "run_with_runtime_budget", fake_bounded_run)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("nightly-warehouse must use run_with_runtime_budget, not subprocess.run")
    ))

    rc = module.run_task("nightly-warehouse", tallanto_phone_limit=1)

    assert rc != 0
    assert latest_path.read_text(encoding="utf-8") == "OLD-LATEST-BYTES"


def test_nightly_self_heal_fails_loud_without_staging_db(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    base_config = staging_root / "nightly_service/base.json"
    base_config.parent.mkdir(parents=True)
    base_config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    monkeypatch.setattr(module, "NIGHTLY_DV2_CONFIG", staging_root / "nightly_service/dv2.json")
    monkeypatch.setattr(module, "NIGHTLY_BASE_CONFIG", base_config)

    reason = module.ensure_nightly_config()

    assert "staging DB is missing" in reason
    assert not module.NIGHTLY_DV2_CONFIG.exists()


def test_nightly_refreshes_wappi_pair_snapshots_even_when_config_is_valid(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    staging_root.mkdir(parents=True)
    timeline_db = staging_root / "customer_timeline_staging.sqlite"
    with sqlite3.connect(timeline_db):
        pass
    config = staging_root / "nightly_service/dv2.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(valid_nightly_payload(staging_root)), encoding="utf-8")
    calls: list[Path] = []

    current_snapshot = module.snapshot_wappi_pair_inputs

    def snapshot(state_root: Path) -> dict:
        calls.append(state_root)
        return current_snapshot(state_root)

    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", timeline_db)
    monkeypatch.setattr(module, "NIGHTLY_DV2_CONFIG", config)
    monkeypatch.setattr(module, "snapshot_wappi_pair_inputs", snapshot)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("valid config must not rebuild")),
    )

    assert module.ensure_nightly_config() == ""
    assert calls == [staging_root / "state"]


def test_nightly_rebuilds_config_when_wappi_pair_inputs_change(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    staging_root.mkdir(parents=True)
    timeline_db = staging_root / "customer_timeline_staging.sqlite"
    payload = valid_nightly_payload(staging_root)
    config = staging_root / "nightly_service/dv2.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(payload), encoding="utf-8")
    changed_sha = hashlib.sha256(b'[{"changed":true}]\n').hexdigest()
    changed_pairs = builder.wappi_pair_snapshot_paths(
        staging_root / "state",
        pairs_sha256=changed_sha,
        auto_pairs_sha256=changed_sha,
    )
    changed_pairs[0].parent.mkdir(parents=True, exist_ok=True)
    for path in changed_pairs:
        path.write_text('[{"changed":true}]\n', encoding="utf-8")

    def changed_snapshot(_state_root: Path) -> dict:
        return {"pairs_file": changed_pairs[0], "auto_pairs_file": changed_pairs[1]}

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        rebuilt = valid_nightly_payload(staging_root)
        wappi = config_step(rebuilt, "wappi_history_incremental")["config"]
        wappi.update(
            pairs_file=str(changed_pairs[0]),
            auto_pairs_file=str(changed_pairs[1]),
            pairs_file_sha256=changed_sha,
            auto_pairs_file_sha256=changed_sha,
        )
        config.write_text(json.dumps(rebuilt), encoding="utf-8")
        return module.subprocess.CompletedProcess(command, 0, stdout="{}")

    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", timeline_db)
    monkeypatch.setattr(module, "NIGHTLY_DV2_CONFIG", config)
    monkeypatch.setattr(module, "NIGHTLY_BASE_CONFIG", staging_root / "nightly_service/base.json")
    monkeypatch.setattr(module, "snapshot_wappi_pair_inputs", changed_snapshot)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.ensure_nightly_config() == ""
    assert len(calls) == 1
    assert config_step(json.loads(config.read_text()), "wappi_history_incremental")["config"][
        "auto_pairs_file_sha256"
    ] == changed_sha


def test_nightly_self_heal_can_rebuild_without_optional_base_config(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    staging_root.mkdir(parents=True)
    timeline_db = staging_root / "customer_timeline_staging.sqlite"
    with sqlite3.connect(timeline_db):
        pass
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", timeline_db)
    monkeypatch.setattr(module, "NIGHTLY_DV2_CONFIG", staging_root / "nightly_service/dv2.json")
    monkeypatch.setattr(module, "NIGHTLY_BASE_CONFIG", staging_root / "nightly_service/base.json")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        module.NIGHTLY_DV2_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        module.NIGHTLY_DV2_CONFIG.write_text(
            json.dumps(valid_nightly_payload(staging_root)), encoding="utf-8"
        )
        return module.subprocess.CompletedProcess(command, 0, stdout="{}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    reason = module.ensure_nightly_config()

    assert reason == ""
    assert len(calls) == 1


def test_nightly_config_rejects_stale_schema_version(tmp_path, monkeypatch) -> None:
    """B1: a config predating config_schema_version (or stamped with an old
    one) must not pass preflight just because every other field happens to
    still be correct."""
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    del payload["config_schema_version"]
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    reason = module.validate_nightly_config(config)

    assert "schema version" in reason


def test_nightly_config_rejects_old_split_state_layout(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(
        module,
        "STAGING_TIMELINE_DB",
        staging_root / "customer_timeline_staging.sqlite",
    )
    payload = valid_nightly_payload(staging_root)
    payload.pop("state_root")
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert "single persistent state tree" in module.validate_nightly_config(config)


def test_nightly_config_rejects_missing_required_manifest_sources(tmp_path, monkeypatch) -> None:
    """B1: this is the exact bug report -- an old nightly.json written before
    required_manifest_sources existed (so the key is simply absent) must be
    rejected, not silently treated as "no required sources" and pass."""
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    del payload["required_manifest_sources"]
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    reason = module.validate_nightly_config(config)

    assert "required_manifest_sources" in reason


def test_nightly_config_rejects_incomplete_required_manifest_sources(tmp_path, monkeypatch) -> None:
    """B1: the check is an *exact* set match, not "at least these" -- a
    config missing even one of the 10 mandatory labels fails."""
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    payload["required_manifest_sources"] = [
        label for label in payload["required_manifest_sources"] if label != "wappi_max"
    ]
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    reason = module.validate_nightly_config(config)

    assert "wappi_max" in reason


def test_nightly_config_v6_rejects_missing_runtime_contract_fields(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    base = valid_nightly_payload(staging_root)
    config = tmp_path / "nightly.json"

    cases = (
        ("AMO page budget", lambda payload: config_step(payload, "amo_incremental_shadow")["config"].update(max_pages=20), "200 pages"),
        ("AMO page size", lambda payload: config_step(payload, "amo_incremental_shadow")["config"].update(page_limit=50), "page_limit"),
        ("Wappi checkpoint", lambda payload: config_step(payload, "wappi_history_incremental")["config"].pop("checkpoint_dir"), "checkpoint"),
        ("Wappi strict nightly", lambda payload: config_step(payload, "wappi_history_incremental")["config"].update(require_widget_linkage=True), "quarantine"),
        ("Wappi old fingerprint", lambda payload: config_step(payload, "wappi_history_incremental")["config"].update(messages_per_chat=100), "transferred checkpoint"),
        ("Wappi partial history", lambda payload: config_step(payload, "wappi_history_incremental")["config"].update(complete_message_history=False), "transferred checkpoint"),
        ("Wappi live pairs", lambda payload: config_step(payload, "wappi_history_incremental")["config"].update(auto_pairs_file=str(tmp_path / "live.json")), "content-addressed staging snapshot"),
        ("Wappi pair SHA", lambda payload: config_step(payload, "wappi_history_incremental")["config"].pop("pairs_file_sha256"), "missing or invalid"),
        ("Wappi phase1 SHA pin", lambda payload: config_step(payload, "wappi_history_incremental")["config"].pop("phase1_config_sha256"), "phase1_config_sha256"),
        ("Wappi phase1 SHA mismatch", lambda payload: config_step(payload, "wappi_history_incremental")["config"].update(phase1_config_sha256="0" * 64), "phase1_config SHA mismatch"),
        ("mail proof SHA", lambda payload: config_step(payload, "mail_archive_incremental")["config"]["sources"][0].pop("proof_manifest_sha256"), "proof_manifest_sha256"),
        ("mail pending reconsider", lambda payload: config_step(payload, "mail_link_enrich")["config"].pop("reconsider_pending"), "reconsider pending"),
        ("mail before Tallanto", lambda payload: move_config_step_before(payload, "mail_link_enrich", "tallanto_cards_sync"), "cards before mail"),
        ("mail identity", lambda payload: config_step(payload, "mail_link_enrich")["config"].update(tallanto_identity_dbs=[str(tmp_path / "missing.sqlite")]), "identity DBs"),
            ("Tallanto cards page budget", lambda payload: config_step(payload, "tallanto_cards_sync")["config"].update(max_pages=20), "500 pages"),
            ("Tallanto money timeout", lambda payload: config_step(payload, "tallanto_money_api_incremental")["config"].pop("timeout_seconds"), "90-minute"),
    )
    for _name, mutate, expected in cases:
        payload = json.loads(json.dumps(base))
        mutate(payload)
        config.write_text(json.dumps(payload), encoding="utf-8")
        assert expected in module.validate_nightly_config(config)


def test_nightly_config_defers_checkpoint_state_to_verified_writer_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    checkpoint = staging_root / "state/wappi/checkpoint/wappi_history_checkpoint.json"
    checkpoint.unlink()
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    # The wrapper validates only immutable config inputs. The importer enforces
    # checkpoint compatibility before network access, using verified ownership
    # passed by the locked nightly service rather than trusting a marker file.
    assert module.validate_nightly_config(config) == ""


def test_nightly_self_heal_rebuilds_stale_on_disk_config(tmp_path, monkeypatch) -> None:
    """B1: ensure_nightly_config() self-heals a config that already exists on
    disk but is stale (old schema version / missing required_manifest_sources)
    -- no manual deletion needed, matching how it already self-heals a
    missing config."""
    staging_root = tmp_path / ".codex_local/staging"
    staging_root.mkdir(parents=True)
    timeline_db = staging_root / "customer_timeline_staging.sqlite"
    with sqlite3.connect(timeline_db):
        pass
    dv2_config = staging_root / "nightly_service/dv2.json"
    dv2_config.parent.mkdir(parents=True)
    stale_payload = valid_nightly_payload(staging_root)
    del stale_payload["config_schema_version"]
    del stale_payload["required_manifest_sources"]
    dv2_config.write_text(json.dumps(stale_payload), encoding="utf-8")
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", timeline_db)
    monkeypatch.setattr(module, "NIGHTLY_DV2_CONFIG", dv2_config)
    monkeypatch.setattr(module, "NIGHTLY_BASE_CONFIG", staging_root / "nightly_service/base.json")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        dv2_config.write_text(json.dumps(valid_nightly_payload(staging_root)), encoding="utf-8")
        return module.subprocess.CompletedProcess(command, 0, stdout="{}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    reason_before = module.validate_nightly_config(dv2_config)
    reason_after = module.ensure_nightly_config()

    assert "schema version" in reason_before
    assert reason_after == ""
    assert len(calls) == 1
    assert module.validate_nightly_config(dv2_config) == ""


def test_nightly_self_heal_rebuilds_and_validates_persistent_config(tmp_path, monkeypatch) -> None:
    nightly_home = tmp_path / "nightly-home"
    staging_root = nightly_home / ".codex_local/staging"
    timeline_db = staging_root / "customer_timeline_staging.sqlite"
    base_config = staging_root / "nightly_service/base.json"
    dv2_config = staging_root / "nightly_service/dv2.json"
    base_config.parent.mkdir(parents=True)
    with sqlite3.connect(timeline_db):
        pass
    base_config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "NIGHTLY_HOME", nightly_home)
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", timeline_db)
    monkeypatch.setattr(module, "NIGHTLY_DV2_CONFIG", dv2_config)
    monkeypatch.setattr(module, "NIGHTLY_BASE_CONFIG", base_config)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        dv2_config.write_text(json.dumps(valid_nightly_payload(staging_root)), encoding="utf-8")
        return module.subprocess.CompletedProcess(command, 0, stdout="{}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    reason = module.ensure_nightly_config()

    assert reason == ""
    command, kwargs = calls[0]
    assert command[command.index("--timeline-db") + 1] == str(timeline_db)
    assert command[command.index("--base-service-config") + 1] == str(base_config)
    state_root = staging_root / "state"
    assert command[command.index("--state-root") + 1] == str(state_root)
    assert command[command.index("--out-root") + 1] == str(state_root)
    assert "nightly_dv2_sources" not in command
    assert kwargs["env"]["CUSTOMER_TIMELINE_NIGHTLY_HOME"] == str(nightly_home)


def test_builder_reuses_verified_mail_process_output_without_rebuilding(tmp_path, monkeypatch) -> None:
    timeline_db = tmp_path / "staging" / "customer_timeline_staging.sqlite"
    timeline_db.parent.mkdir(parents=True)
    state_root = timeline_db.parent / "state"
    mail_root = state_root / "mail_pipeline"
    process_root = mail_root / "process"
    process_root.mkdir(parents=True)
    output_jsonl = process_root / "mail_archive_stage2_incremental.jsonl"
    builder_manifest = process_root / "mail_increment_builder_manifest.json"
    output_jsonl.write_text('{"source_id":"mail-1"}\n', encoding="utf-8")
    builder_manifest.write_text('{"rows_written":1}\n', encoding="utf-8")
    process_manifest = mail_root / "mail_process_manifest.json"
    process_manifest.write_text(
        json.dumps(
            {
                "status": "ok",
                "finished_at": "2026-08-28T12:00:00+00:00",
                "rows_written": 1,
                "max_event_at": "2026-08-28T12:00:00+00:00",
                "output_jsonl": str(output_jsonl),
                "output_sha256": builder.sha256_file(output_jsonl),
                "builder_manifest": str(builder_manifest),
                "builder_manifest_sha256": builder.sha256_file(builder_manifest),
            }
        ),
        encoding="utf-8",
    )
    output_before = output_jsonl.read_bytes()
    manifest_before = builder_manifest.read_bytes()
    captured = {}

    def fake_service_config(**kwargs):
        captured["state_root"] = kwargs["state_root"]
        captured["mail_jsonl"] = kwargs["mail_jsonl"]
        captured["mail_process_manifest"] = kwargs["mail_process_manifest"]
        captured["mail_process_manifest_sha256"] = kwargs["mail_process_manifest_sha256"]
        return {}

    monkeypatch.setattr(
        builder,
        "build_mail_increment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("builder must not produce mail")),
    )
    monkeypatch.setattr(builder, "build_mango_freshness", lambda *args, **kwargs: {})
    monkeypatch.setattr(builder, "build_service_config", fake_service_config)

    result = builder.main(
        [
            "--source-root",
            str(tmp_path / "source"),
            "--mail-data-root",
            str(tmp_path / "mail"),
            "--out-root",
            str(state_root),
            "--state-root",
            str(state_root),
            "--timeline-db",
            str(timeline_db),
        ]
    )

    assert result == 0
    assert captured["state_root"] == state_root.resolve()
    assert captured["mail_jsonl"] == output_jsonl
    assert captured["mail_process_manifest"] == process_manifest
    assert captured["mail_process_manifest_sha256"] == builder.sha256_file(process_manifest)
    assert output_jsonl.read_bytes() == output_before
    assert builder_manifest.read_bytes() == manifest_before


def test_wappi_pair_inputs_are_private_frozen_snapshots(tmp_path: Path, monkeypatch) -> None:
    manual_source = tmp_path / "live/manual.json"
    auto_source = tmp_path / "live/auto.json"
    manual_source.parent.mkdir(parents=True)
    manual_source.write_text("[]\n", encoding="utf-8")
    auto_source.write_text('{"pairs": []}\n', encoding="utf-8")
    monkeypatch.setattr(builder, "DEFAULT_WAPPI_PAIRS", manual_source)
    monkeypatch.setattr(builder, "DEFAULT_WAPPI_AUTO_PAIRS", auto_source)
    state_root = tmp_path / "staging/state"

    first = REAL_SNAPSHOT_WAPPI_PAIR_INPUTS(state_root)
    auto_snapshot = Path(first["auto_pairs_file"])
    frozen_bytes = auto_snapshot.read_bytes()
    auto_source.write_text("[]\n", encoding="utf-8")

    assert auto_snapshot.read_bytes() == frozen_bytes
    assert auto_snapshot.stat().st_mode & 0o777 == 0o400
    assert Path(first["manifest"]).stat().st_mode & 0o777 == 0o600
    second = REAL_SNAPSHOT_WAPPI_PAIR_INPUTS(state_root)
    assert Path(second["auto_pairs_file"]) != auto_snapshot
    assert auto_snapshot.read_bytes() == frozen_bytes
    assert Path(second["auto_pairs_file"]).read_bytes() == auto_source.read_bytes()


def test_wappi_pair_snapshot_rejects_source_change_during_copy(tmp_path: Path, monkeypatch) -> None:
    manual_source = tmp_path / "live/manual.json"
    auto_source = tmp_path / "live/auto.json"
    manual_source.parent.mkdir(parents=True)
    manual_source.write_text('{"pairs": []}\n', encoding="utf-8")
    auto_source.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(builder, "DEFAULT_WAPPI_PAIRS", manual_source)
    monkeypatch.setattr(builder, "DEFAULT_WAPPI_AUTO_PAIRS", auto_source)

    def changed_copy(_source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(builder, "atomic_publish_latest", changed_copy)

    with pytest.raises(RuntimeError, match="changed while creating"):
        REAL_SNAPSHOT_WAPPI_PAIR_INPUTS(tmp_path / "staging/state")


def test_wappi_pair_snapshot_keeps_concurrent_winner(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.json"
    target = tmp_path / "snapshots/manual.json"
    source.write_text("[]\n", encoding="utf-8")
    winner = b'{"pairs": []}\n'

    def raced_link(_candidate: Path, destination: Path) -> None:
        destination.write_bytes(winner)
        raise FileExistsError

    monkeypatch.setattr(builder.os, "link", raced_link)

    assert builder._publish_content_addressed_input(source, target) is False
    assert target.read_bytes() == winner


def test_builder_mail_process_input_is_missing_or_sha_verified(tmp_path) -> None:
    state_root = tmp_path / "staging/state"
    missing = builder.resolve_mail_process_input(state_root)

    assert missing["status"] == "missing"
    assert missing["verified"] is False
    assert not (state_root / "mail_pipeline").exists()

    mail_root = state_root / "mail_pipeline"
    process_root = mail_root / "process"
    process_root.mkdir(parents=True)
    output_jsonl = process_root / "mail_archive_stage2_incremental.jsonl"
    builder_manifest = process_root / "mail_increment_builder_manifest.json"
    output_jsonl.write_text("{}\n", encoding="utf-8")
    builder_manifest.write_text("{}\n", encoding="utf-8")
    (mail_root / "mail_process_manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "finished_at": "2026-09-02T00:00:00+00:00",
                "rows_written": 1,
                "max_event_at": "2026-09-02T00:00:00+00:00",
                "output_jsonl": str(output_jsonl),
                "output_sha256": "wrong",
                "builder_manifest": str(builder_manifest),
                "builder_manifest_sha256": builder.sha256_file(builder_manifest),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="mail_process_output_sha_mismatch"):
        builder.resolve_mail_process_input(state_root)


def test_builder_rejects_mail_process_manifest_without_sha_pin(tmp_path) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    state_root = staging_root / "state"
    mail_data_root = _mail_root_with_identity(tmp_path)
    mail_jsonl, mail_manifest, _mail_manifest_sha = _write_verified_mail_process_state(state_root)

    with pytest.raises(RuntimeError, match="mail_process_manifest_sha256_missing"):
        builder.build_service_config(
            timeline_db=staging_root / "customer_timeline_staging.sqlite",
            state_root=state_root,
            mail_jsonl=mail_jsonl,
            mail_process_manifest=mail_manifest,
            mail_process_manifest_sha256=None,
            mango_manifest=state_root / "freshness/mango.json",
            base_service_config=staging_root / "missing.json",
            mail_data_root=mail_data_root,
        )


def test_builder_uses_measured_mango_freshness_cursor(tmp_path) -> None:
    source_root = tmp_path / "source"
    capture = source_root / "product_data/mango_update_after_202607_test"
    capture.mkdir(parents=True)
    manifest = tmp_path / "mango_freshness.json"

    report = builder.build_mango_freshness(source_root, manifest)

    expected = report["latest"]["mtime_utc"]
    assert builder.freshness_cursor_from_manifest(manifest) == expected
    assert expected != "1970-01-01T00:00:00+00:00"


def test_builder_rejects_second_descendant_state_tree(tmp_path) -> None:
    timeline_db = tmp_path / "staging/customer_timeline_staging.sqlite"
    timeline_db.parent.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="single <staging>/state tree"):
        builder.main(
            [
                "--out-root",
                str(timeline_db.parent / "state/other"),
                "--state-root",
                str(timeline_db.parent / "state/other"),
                "--timeline-db",
                str(timeline_db),
            ]
        )


def test_builder_reuses_single_existing_state_tree(tmp_path) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    state_root = staging_root / "state"
    tasks_snapshot = state_root / "amo_tasks/amo_tasks_snapshot.csv"
    tasks_snapshot.parent.mkdir(parents=True)
    tasks_snapshot.write_text("task_id\n", encoding="utf-8")
    mail_data_root = _mail_root_with_identity(tmp_path)
    mail_jsonl, mail_manifest, mail_manifest_sha = _write_verified_mail_process_state(state_root)

    payload = builder.build_service_config(
        timeline_db=staging_root / "customer_timeline_staging.sqlite",
        state_root=state_root,
        mail_jsonl=mail_jsonl,
        mail_process_manifest=mail_manifest,
        mail_process_manifest_sha256=mail_manifest_sha,
        mango_manifest=state_root / "freshness/mango.json",
        base_service_config=staging_root / "missing.json",
        mail_data_root=mail_data_root,
        amo_tasks_snapshot=builder.resolve_amo_tasks_snapshot(state_root, explicit=None),
    )

    steps = {step["name"]: step for step in payload["steps"]}
    assert steps["mango_processed_sweep"]["config"]["out_jsonl"] == str(
        state_root / "calls/mango_processed_sweep.jsonl"
    )
    assert steps["calls_and_amo_incremental"]["config"]["journal_path"] == str(
        state_root / "calls/import_journal.jsonl"
    )
    assert steps["amo_incremental_shadow"]["config"]["out_root"] == str(
        state_root / "amo_incremental"
    )
    assert steps["amo_incremental_shadow"]["config"]["tasks_snapshot"] == str(tasks_snapshot)
    assert steps["wappi_history_incremental"]["config"]["widget_link_db"] == str(
        state_root / "wappi/wappi_amo_links.sqlite"
    )
    assert steps["wappi_history_incremental"]["config"]["checkpoint_dir"] == str(
        state_root / "wappi/checkpoint"
    )
    assert steps["mail_archive_incremental"]["config"]["journal_path"] == str(
        state_root / "mail_pipeline/process/mail_incremental_journal.jsonl"
    )
    mail_source = steps["mail_archive_incremental"]["config"]["sources"][0]
    assert mail_source["proof_manifest_path"] == str(
        state_root / "mail_pipeline/mail_process_manifest.json"
    )
    assert mail_source["proof_max_age_hours"] == 72.0
    assert steps["mail_link_enrich"]["config"]["out_dir"] == str(
        state_root / "mail_pipeline/mail_link_enrich"
    )
    assert steps["tallanto_cards_sync"]["config"]["out_root"] == str(
        state_root / "tallanto_cards"
    )
    assert payload["out_root"] == str(state_root / "nightly_service_runs")
    assert steps["wappi_history_incremental"]["config"]["phase1_config_sha256"] == builder.sha256_file(
        builder.DEFAULT_WAPPI_CONFIG
    )
    assert "nightly_dv2_sources" not in json.dumps(payload)


def test_builder_creates_calls_step_without_optional_base_config(tmp_path) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    mail_data_root = _mail_root_with_identity(tmp_path)
    mail_jsonl, mail_manifest, mail_manifest_sha = _write_verified_mail_process_state(
        staging_root / "state"
    )

    payload = builder.build_service_config(
        timeline_db=staging_root / "customer_timeline_staging.sqlite",
        state_root=staging_root / "state",
        mail_jsonl=mail_jsonl,
        mail_process_manifest=mail_manifest,
        mail_process_manifest_sha256=mail_manifest_sha,
        mango_manifest=staging_root / "state/freshness/mango.json",
        base_service_config=staging_root / "missing.json",
        mail_data_root=mail_data_root,
    )

    steps = {step["name"]: step for step in payload["steps"]}
    sources = steps["calls_and_amo_incremental"]["config"]["sources"]
    assert [source["source_system"] for source in sources] == ["mango_processed_summary"]
    amo = steps["amo_incremental_shadow"]
    assert amo["kind"] == "amo_incremental"
    assert amo["required"] is True
    assert amo["config"]["page_limit"] == 20
    assert amo["config"]["max_pages"] == 200
    assert amo["config"]["timeline_db"] == str(staging_root / "customer_timeline_staging.sqlite")
    wappi = steps["wappi_history_incremental"]
    assert wappi["config"]["require_widget_linkage"] is False
    assert wappi["config"]["phase1_config_sha256"] == builder.sha256_file(builder.DEFAULT_WAPPI_CONFIG)
    expected_pairs, expected_auto_pairs = builder.wappi_pair_snapshot_paths(
        staging_root / "state",
        pairs_sha256=wappi["config"]["pairs_file_sha256"],
        auto_pairs_sha256=wappi["config"]["auto_pairs_file_sha256"],
    )
    assert wappi["config"]["pairs_file"] == str(expected_pairs)
    assert wappi["config"]["auto_pairs_file"] == str(expected_auto_pairs)
    assert wappi["config"]["pairs_file_sha256"] == builder.sha256_file(expected_pairs)
    assert wappi["config"]["auto_pairs_file_sha256"] == builder.sha256_file(expected_auto_pairs)
    assert wappi["config"]["messages_per_chat"] == 50000
    assert wappi["config"]["complete_message_history"] is True
    assert wappi["config"]["checkpoint_dir"] == str(
        staging_root / "state/wappi/checkpoint"
    )
    cards = steps["tallanto_cards_sync"]
    assert cards["kind"] == "tallanto_cards"
    assert cards["required"] is True
    assert list(steps).index("tallanto_cards_sync") < list(steps).index("tallanto_attendance_api_incremental")
    assert list(steps).index("tallanto_attendance_api_incremental") < list(steps).index("tallanto_money_api_incremental")
    assert list(steps).index("tallanto_money_api_incremental") < list(steps).index("wappi_history_incremental")
    attendance_api = steps["tallanto_attendance_api_incremental"]
    assert attendance_api["kind"] == "tallanto_attendance_api"
    assert attendance_api["required"] is True
    assert attendance_api["config"]["apply"] is True
    assert attendance_api["config"]["initial_since"] == "2026-06-09T00:00:00+03:00"
    money_api = steps["tallanto_money_api_incremental"]
    assert money_api["kind"] == "tallanto_money_api"
    assert money_api["required"] is True
    assert money_api["config"]["apply"] is True
    assert money_api["config"]["timeout_seconds"] == 5400
    assert money_api["config"]["timeline_db"] == str(
        staging_root / "customer_timeline_staging.sqlite"
    )
    assert "tallanto_money_incremental" not in steps
    mail_link_enrich = steps["mail_link_enrich"]
    assert mail_link_enrich["required"] is True
    assert list(steps).index("tallanto_cards_sync") < list(steps).index("mail_link_enrich")
    assert mail_link_enrich["config"]["reconsider_pending"] is True
    assert mail_link_enrich["config"]["tallanto_identity_dbs"] == [
        str(mail_data_root / CANONICAL_MAIL_IDENTITY_DB)
    ]
    assert ".codex_local/staging" not in " ".join(mail_link_enrich["config"]["tallanto_identity_dbs"])
    stage4b = steps["stage4b_bot_opening"]
    assert stage4b["required"] is True
    assert stage4b["config"]["defer_full_db_check"] is True
    assert list(steps).index("derived_signals_refresh") < list(steps).index("stage4b_bot_opening")
    assert list(steps).index("stage4b_bot_opening") < list(steps).index("bot_safe_rebuild")


def test_builder_uses_explicit_existing_tallanto_identity_db_without_mail_root(tmp_path) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    identity_db = tmp_path / "readonly/tallanto_identity.sqlite"
    identity_db.parent.mkdir(parents=True)
    identity_db.touch()
    mail_jsonl, mail_manifest, mail_manifest_sha = _write_verified_mail_process_state(
        staging_root / "state"
    )

    payload = builder.build_service_config(
        timeline_db=staging_root / "customer_timeline_staging.sqlite",
        state_root=staging_root / "state",
        mail_jsonl=mail_jsonl,
        mail_process_manifest=mail_manifest,
        mail_process_manifest_sha256=mail_manifest_sha,
        mango_manifest=staging_root / "state/freshness/mango.json",
        base_service_config=staging_root / "missing.json",
        mail_data_root=tmp_path / "missing-mail-root",
        tallanto_identity_dbs=[identity_db],
    )

    steps = {step["name"]: step for step in payload["steps"]}
    assert steps["mail_link_enrich"]["config"]["tallanto_identity_dbs"] == [
        str(identity_db.resolve())
    ]

    with pytest.raises(FileNotFoundError, match="Tallanto identity DB is missing"):
        builder.resolve_tallanto_identity_dbs(
            tmp_path / "missing-mail-root",
            explicit=[tmp_path / "missing.sqlite"],
        )


def test_builder_accepts_base_calls_step_without_optional_amo_sources(tmp_path) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    mail_data_root = _mail_root_with_identity(tmp_path)
    mail_jsonl, mail_manifest, mail_manifest_sha = _write_verified_mail_process_state(
        staging_root / "state"
    )
    base = staging_root / "nightly_service/base.json"
    base.parent.mkdir(parents=True)
    base.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "name": "calls_and_amo_incremental",
                        "config": {
                            "journal_path": str(staging_root / "nightly_service/calls.jsonl"),
                            "sources": [
                                {
                                    "source_system": "mango_processed_summary",
                                    "normalizer": "mango_processed_summary",
                                    "required": True,
                                    "path": str(staging_root / "mango.jsonl"),
                                },
                                {
                                    "source_system": "amocrm_snapshot",
                                    "normalizer": "amo_snapshot",
                                    "required": True,
                                    "path": str(staging_root / "legacy_amo.jsonl"),
                                },
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = builder.build_service_config(
        timeline_db=staging_root / "customer_timeline_staging.sqlite",
        state_root=staging_root / "state",
        mail_jsonl=mail_jsonl,
        mail_process_manifest=mail_manifest,
        mail_process_manifest_sha256=mail_manifest_sha,
        mango_manifest=staging_root / "state/freshness/mango.json",
        base_service_config=base,
        mail_data_root=mail_data_root,
    )

    sources = next(
        step for step in payload["steps"] if step["name"] == "calls_and_amo_incremental"
    )["config"]["sources"]
    assert [source["source_system"] for source in sources] == ["mango_processed_summary"]


def test_builder_keeps_required_calls_mail_and_sweep_steps(tmp_path) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    mail_data_root = _mail_root_with_identity(tmp_path)
    mail_jsonl, mail_manifest, mail_manifest_sha = _write_verified_mail_process_state(
        staging_root / "state"
    )
    base = staging_root / "nightly_service/base.json"
    base.parent.mkdir(parents=True)
    base.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "name": "calls_and_amo_incremental",
                        "kind": "nightly_incremental",
                        "enabled": True,
                        "required": True,
                        "config": {
                            "journal_path": str(staging_root / "nightly_service/calls_and_amo.jsonl"),
                            "sources": [
                                {
                                    "name": "old_mango",
                                    "source_system": "mango_processed_summary",
                                    "path": str(staging_root / "old.jsonl"),
                                    "normalizer": "mango_processed_summary",
                                    "required": True,
                                },
                                *[
                                    {
                                        "name": source_system,
                                        "source_system": source_system,
                                        "path": str(staging_root / f"{source_system}.jsonl"),
                                        "normalizer": normalizer,
                                        "required": True,
                                    }
                                    for source_system, normalizer in builder.REQUIRED_CALL_SOURCES.items()
                                    if source_system != "mango_processed_summary"
                                ],
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = builder.build_service_config(
        timeline_db=staging_root / "customer_timeline_staging.sqlite",
        state_root=staging_root / "state",
        mail_jsonl=mail_jsonl,
        mail_process_manifest=mail_manifest,
        mail_process_manifest_sha256=mail_manifest_sha,
        mango_manifest=staging_root / "state/freshness/mango.json",
        base_service_config=base,
        mail_data_root=mail_data_root,
    )

    steps = {step["name"]: step for step in payload["steps"]}
    assert {"mango_processed_sweep", "calls_and_amo_incremental", "mail_archive_incremental"} <= steps.keys()
    assert all(steps[name]["required"] is True for name in module.REQUIRED_NIGHTLY_STEPS)
    assert steps["mango_processed_sweep"]["config"]["package_dbs"] == [
        str(builder.configured_calls_working_db(builder.DEFAULT_MANGO_CALLS_SERVICE_CONFIG))
    ]
    assert steps["mango_processed_sweep"]["config"]["scan_roots"] == []
    mango_source = steps["calls_and_amo_incremental"]["config"]["sources"][0]
    assert mango_source["path"].endswith("state/calls/mango_processed_sweep.jsonl")


def test_summary_is_exactly_five_lines(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "FOTON_DAILY", tmp_path)
    now = module.datetime.now(module.timezone.utc)

    summary = module.write_summary(
        task="mail-capture",
        started=now,
        finished=now,
        command=("bash", "scripts/run_customer_timeline_mail_capture_daily.sh", "--apply"),
        log_path=tmp_path / "task.log",
        rc=0,
        status="ok",
        stop_reason="",
        metrics="rows_written=1",
        expected_output=tmp_path / "manifest.json",
    )

    assert len(summary.read_text(encoding="utf-8").splitlines()) == 5


def test_summary_omits_extra_metrics_when_not_provided(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "FOTON_DAILY", tmp_path)
    now = module.datetime.now(module.timezone.utc)

    summary = module.write_summary(
        task="mail-capture",
        started=now,
        finished=now,
        command=("bash", "scripts/run_customer_timeline_mail_capture_daily.sh", "--apply"),
        log_path=tmp_path / "task.log",
        rc=0,
        status="ok",
        stop_reason="",
        metrics="rows_written=1",
        expected_output=tmp_path / "manifest.json",
    )

    assert "prod_snapshot_staleness" not in summary.read_text(encoding="utf-8")


def test_task_success_age_is_persistent_and_failed_run_keeps_last_success(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "TASK_STATE_ROOT", tmp_path)
    now = module.datetime(2026, 7, 13, 12, 0, tzinfo=module.timezone.utc)

    first = module.task_success_age_metric("mail-download", status="ok", finished=now)
    failed_later = module.task_success_age_metric(
        "mail-download",
        status="failed",
        finished=now + timedelta(hours=31),
    )

    assert "last_success_status=ok" in first
    assert "last_success_age_hours=31.0" in failed_later
    assert "last_success_status=alert" in failed_later


def test_prod_snapshot_staleness_metric_marks_fresh_snapshot_ok(tmp_path, monkeypatch) -> None:
    db = tmp_path / "customer_timeline.sqlite"
    db.write_text("sqlite", encoding="utf-8")
    now = module.datetime(2026, 7, 10, 12, 0, tzinfo=module.timezone.utc)
    os.utime(db, (now.timestamp(), now.timestamp()))
    monkeypatch.setattr(module, "PROD_TIMELINE_DB", db)

    metric = module.prod_snapshot_staleness_metric(now)

    assert "prod_snapshot_staleness=ok" in metric


def test_prod_snapshot_staleness_metric_marks_old_snapshot_alert(tmp_path, monkeypatch) -> None:
    db = tmp_path / "customer_timeline.sqlite"
    db.write_text("sqlite", encoding="utf-8")
    now = module.datetime(2026, 7, 10, 12, 0, tzinfo=module.timezone.utc)
    old = now - timedelta(days=8)
    os.utime(db, (old.timestamp(), old.timestamp()))
    monkeypatch.setattr(module, "PROD_TIMELINE_DB", db)

    metric = module.prod_snapshot_staleness_metric(now)

    assert "prod_snapshot_staleness=alert" in metric
