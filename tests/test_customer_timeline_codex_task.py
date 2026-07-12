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
