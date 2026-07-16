from __future__ import annotations

import importlib.util
import os
import sys
import json
from datetime import timedelta
from pathlib import Path

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


def valid_nightly_payload(staging_root: Path) -> dict:
    return {
        "timeline_db": str(staging_root / "customer_timeline_staging.sqlite"),
        "allowed_root": str(staging_root),
        "steps": [
            {
                "name": "mango_processed_sweep",
                "enabled": True,
                "required": True,
                "config": {
                    "producer_script": str(module.ROOT / "scripts/build_mango_call_timeline_increment.py"),
                    "package_dbs": [str(module.MANGO_READY_PACKAGE_DB)],
                },
            },
            {
                "name": "calls_and_amo_incremental",
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
                        }
                        for source_system, normalizer in module.REQUIRED_CALL_SOURCES.items()
                    ],
                },
            },
            {"name": "mail_archive_incremental", "enabled": True, "required": True},
        ],
    }


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


def test_mail_process_task_requires_fresh_download_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "MAIL_STATE_DIR", tmp_path)

    task = module.build_task_spec("mail-process", tallanto_phone_limit=1)

    assert task.command
    assert "stage manifest is missing" in task.stop_reason


def test_mail_import_task_uses_mail_only_import_wrapper(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "MAIL_STATE_DIR", tmp_path)
    monkeypatch.setattr(module, "current_runtime", lambda: {"head": "abc", "worktree": "tree"})
    (tmp_path / "mail_process_manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "finished_at": module.datetime.now(module.timezone.utc).isoformat(),
                "runtime": {"head": "abc", "worktree": "tree"},
            }
        ),
        encoding="utf-8",
    )

    task = module.build_task_spec("mail-import", tallanto_phone_limit=1)

    assert task.stop_reason == ""
    assert task.command[1:] == (
        "scripts/run_customer_timeline_mail_import.py",
        "--state-dir",
        str(tmp_path),
    )


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


def test_nightly_config_rejects_sweep_without_ready_package_db(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    payload = valid_nightly_payload(staging_root)
    sweep = next(step for step in payload["steps"] if step["name"] == "mango_processed_sweep")
    sweep["config"]["package_dbs"] = []
    config = tmp_path / "nightly.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    reason = module.validate_nightly_config(config)

    assert "mango_calls_ready.sqlite" in reason


def test_nightly_config_rejects_missing_ready_package_db(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    missing_ready = tmp_path / "drop/mango_calls_ready.sqlite"
    monkeypatch.setattr(module, "STAGING_ROOT", staging_root)
    monkeypatch.setattr(module, "STAGING_TIMELINE_DB", staging_root / "customer_timeline_staging.sqlite")
    monkeypatch.setattr(module, "MANGO_READY_PACKAGE_DB", missing_ready)
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


def test_nightly_self_heal_can_rebuild_without_optional_base_config(tmp_path, monkeypatch) -> None:
    staging_root = tmp_path / ".codex_local/staging"
    staging_root.mkdir(parents=True)
    timeline_db = staging_root / "customer_timeline_staging.sqlite"
    timeline_db.write_bytes(b"sqlite")
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


def test_nightly_self_heal_rebuilds_and_validates_persistent_config(tmp_path, monkeypatch) -> None:
    nightly_home = tmp_path / "nightly-home"
    staging_root = nightly_home / ".codex_local/staging"
    timeline_db = staging_root / "customer_timeline_staging.sqlite"
    base_config = staging_root / "nightly_service/base.json"
    dv2_config = staging_root / "nightly_service/dv2.json"
    base_config.parent.mkdir(parents=True)
    timeline_db.write_bytes(b"sqlite")
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
    assert kwargs["env"]["CUSTOMER_TIMELINE_NIGHTLY_HOME"] == str(nightly_home)


def test_builder_creates_calls_step_without_optional_base_config(tmp_path) -> None:
    staging_root = tmp_path / ".codex_local/staging"

    payload = builder.build_service_config(
        timeline_db=staging_root / "customer_timeline_staging.sqlite",
        out_root=staging_root / "nightly_dv2_sources",
        mail_jsonl=staging_root / "mail.jsonl",
        mail_manifest=staging_root / "mail.json",
        mango_manifest=staging_root / "mango.json",
        tallanto_manifest=staging_root / "tallanto.json",
        base_service_config=staging_root / "missing.json",
    )

    steps = {step["name"]: step for step in payload["steps"]}
    sources = steps["calls_and_amo_incremental"]["config"]["sources"]
    assert [source["source_system"] for source in sources] == ["mango_processed_summary"]


def test_builder_accepts_base_calls_step_without_optional_amo_sources(tmp_path) -> None:
    staging_root = tmp_path / ".codex_local/staging"
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
                                }
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
        out_root=staging_root / "nightly_dv2_sources",
        mail_jsonl=staging_root / "mail.jsonl",
        mail_manifest=staging_root / "mail.json",
        mango_manifest=staging_root / "mango.json",
        tallanto_manifest=staging_root / "tallanto.json",
        base_service_config=base,
    )

    sources = next(
        step for step in payload["steps"] if step["name"] == "calls_and_amo_incremental"
    )["config"]["sources"]
    assert [source["source_system"] for source in sources] == ["mango_processed_summary"]


def test_builder_keeps_required_calls_mail_and_sweep_steps(tmp_path) -> None:
    staging_root = tmp_path / ".codex_local/staging"
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
        out_root=staging_root / "nightly_dv2_sources",
        mail_jsonl=staging_root / "mail.jsonl",
        mail_manifest=staging_root / "mail.json",
        mango_manifest=staging_root / "mango.json",
        tallanto_manifest=staging_root / "tallanto.json",
        base_service_config=base,
    )

    steps = {step["name"]: step for step in payload["steps"]}
    assert {"mango_processed_sweep", "calls_and_amo_incremental", "mail_archive_incremental"} <= steps.keys()
    assert all(steps[name]["required"] is True for name in module.REQUIRED_NIGHTLY_STEPS)
    assert steps["mango_processed_sweep"]["config"]["package_dbs"] == [
        str(builder.MANGO_READY_PACKAGE_DB)
    ]
    mango_source = steps["calls_and_amo_incremental"]["config"]["sources"][0]
    assert mango_source["path"].endswith("nightly_dv2_sources/mango_processed_sweep.jsonl")


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
