from __future__ import annotations

import json
import stat
from datetime import date
from pathlib import Path

import pytest

from mango_mvp.productization.ready_publication import (
    commit_ready_generation,
    inspect_ready_publication,
)
from scripts import run_mango_calls_publication_coordinator as coordinator


DAY = date(2026, 8, 10)


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    pipeline = home / ".mango_local" / "pipeline"
    pipeline.mkdir(parents=True)
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "pipeline_root": str(pipeline),
                "publication_root": str(
                    home / ".mango_local" / "mango_calls_publication"
                ),
                "min_free_gib": 1,
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _manifest(*, closed: bool) -> dict[str, object]:
    return {
        "daily_verdicts": {
            DAY.isoformat(): {
                "closure_ok": closed,
                "mango_unique": 4,
                "pending_unique": 4 if not closed else 0,
                "pending_over_sla": 1 if not closed else 0,
                "oldest_pending_age_minutes": 90 if not closed else 0,
            }
        }
    }


def test_daily_status_always_writes_honest_incomplete_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    monkeypatch.setattr(coordinator, "_ready_manifest", lambda *_args: _manifest(closed=False))
    monkeypatch.setattr(
        coordinator,
        "_daily_export",
        lambda *_args, **_kwargs: {
            "package_status": "INCOMPLETE_DO_NOT_USE_AS_FINAL",
            "reused": False,
        },
    )

    result = coordinator.run(config_path, "daily-status", day=DAY)

    assert result["status"] == "incomplete"
    assert result["closure_ok"] is False
    assert result["external_write"] is False
    status_file = Path(str(result["status_file"]))
    assert status_file.is_file()
    assert stat.S_IMODE(status_file.stat().st_mode) == 0o600
    assert json.loads(status_file.read_text(encoding="utf-8"))["package_status"] == (
        "INCOMPLETE_DO_NOT_USE_AS_FINAL"
    )


def test_closed_verdict_cannot_hide_incomplete_export_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        coordinator, "_ready_manifest", lambda *_args: _manifest(closed=True)
    )
    monkeypatch.setattr(
        coordinator,
        "_daily_export",
        lambda *_args, **_kwargs: {
            "package_status": "INCOMPLETE_DO_NOT_USE_AS_FINAL",
            "closure_ok": False,
            "ready_rows": 0,
            "reused": False,
        },
    )

    close = coordinator.run(config_path, "daily-close", day=DAY)
    status = coordinator.run(config_path, "daily-status", day=DAY)

    assert close["status"] == "incomplete"
    assert close["target_status"] == "incomplete"
    assert close["attempts"][0]["closure_ok"] is False
    assert status["status"] == "incomplete"
    assert status["closure_ok"] is False
    assert status["package_status"] == "INCOMPLETE_DO_NOT_USE_AS_FINAL"


def test_daily_status_records_export_failure_without_leaking_exception_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    monkeypatch.setattr(coordinator, "_ready_manifest", lambda *_args: _manifest(closed=False))

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("+79990001122 /Users/private/secret.sqlite")

    monkeypatch.setattr(coordinator, "_daily_export", fail)

    result = coordinator.run(config_path, "daily-status", day=DAY)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert "+7999" not in serialized and "/Users/" not in serialized
    assert Path(str(result["status_file"])).is_file()


def test_daily_alert_is_aggregate_only_and_flags_sla_and_all_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    monkeypatch.setattr(coordinator, "_ready_manifest", lambda *_args: _manifest(closed=False))
    monkeypatch.setattr(
        coordinator.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 2 * 1024**3})(),
    )

    result = coordinator.run(config_path, "daily-alert", day=DAY)
    serialized = json.dumps(result, ensure_ascii=False).casefold()

    assert result["status"] == "alert"
    assert "pending_over_sla" in result["stop_reason"]
    assert "all_calls_pending_over_60_minutes" in result["stop_reason"]
    assert "phone" not in serialized and "transcript" not in serialized
    assert result["external_write"] is False


def test_daily_close_does_not_export_before_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    monkeypatch.setattr(coordinator, "_ready_manifest", lambda *_args: _manifest(closed=False))
    monkeypatch.setattr(
        coordinator,
        "_daily_export",
        lambda *_args, **_kwargs: pytest.fail("export must not run"),
    )

    result = coordinator.run(config_path, "daily-close", day=DAY)

    assert result["status"] == "pending_closure"
    assert result["external_write"] is False


def test_daily_close_retries_closed_days_from_the_previous_72_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    older = DAY - date.resolution
    manifest = _manifest(closed=False)
    manifest["daily_verdicts"][older.isoformat()] = {  # type: ignore[index]
        "closure_ok": True
    }
    monkeypatch.setattr(coordinator, "_ready_manifest", lambda *_args: manifest)
    exported: list[date] = []

    def export(
        _config: object,
        _root: Path,
        day: date,
        *,
        sealed_only: bool,
        expected_ready_manifest_sha256: str | None = None,
    ) -> dict[str, object]:
        assert sealed_only is True
        assert expected_ready_manifest_sha256 is None
        exported.append(day)
        return {
            "package_status": "FINAL_CLOSED",
            "closure_ok": True,
            "reused": False,
        }

    monkeypatch.setattr(coordinator, "_daily_export", export)

    result = coordinator.run(config_path, "daily-close", day=DAY)

    assert result["status"] == "pending_closure"
    assert exported == [older]
    assert any(
        item["day"] == older.isoformat() and item["status"] == "closed"
        for item in result["attempts"]
    )


def test_daily_close_records_current_decision_sha_when_package_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    decision_sha = "d" * 64
    package_source_sha = "a" * 64
    monkeypatch.setattr(
        coordinator,
        "_ready_snapshot",
        lambda *_args: (_manifest(closed=True), decision_sha),
    )

    def reused_export(
        _config: object,
        _root: Path,
        _day: date,
        *,
        sealed_only: bool,
        expected_ready_manifest_sha256: str | None = None,
    ) -> dict[str, object]:
        assert sealed_only is True
        assert expected_ready_manifest_sha256 == decision_sha
        return {
            "package_status": "FINAL_CLOSED",
            "closure_ok": True,
            "reused": True,
            "source_ready_manifest_sha256": package_source_sha,
        }

    monkeypatch.setattr(coordinator, "_daily_export", reused_export)

    result = coordinator.run(config_path, "daily-close", day=DAY)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    persisted = json.loads(
        (
            Path(str(config["publication_root"]))
            / "state"
            / "daily-close.json"
        ).read_text(encoding="utf-8")
    )

    assert result["status"] == "closed"
    assert result["decision_ready_manifest_sha256"] == decision_sha
    assert persisted["decision_ready_manifest_sha256"] == decision_sha
    assert persisted["attempts"][0][
        "package_source_ready_manifest_sha256"
    ] == package_source_sha
    assert result["attempts"] == [
        {
            "day": DAY.isoformat(),
            "status": "closed",
            "reused": True,
            "package_status": "FINAL_CLOSED",
            "closure_ok": True,
            "package_source_ready_manifest_sha256": package_source_sha,
            "decision_ready_manifest_sha256": decision_sha,
        }
    ]


def test_daily_close_keeps_the_full_72_hour_boundary_reachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    boundary = DAY - 3 * date.resolution
    manifest = _manifest(closed=False)
    manifest["daily_verdicts"][boundary.isoformat()] = {  # type: ignore[index]
        "closure_ok": True
    }
    monkeypatch.setattr(coordinator, "_ready_manifest", lambda *_args: manifest)
    exported: list[date] = []
    monkeypatch.setattr(
        coordinator,
        "_daily_export",
        lambda _config, _root, candidate, *, sealed_only, **_kwargs: (
            exported.append(candidate)
            or {
                "package_status": "FINAL_CLOSED",
                "closure_ok": True,
                "reused": False,
            }
        ),
    )

    result = coordinator.run(config_path, "daily-close", day=DAY)

    assert result["status"] == "pending_closure"
    assert exported == [boundary]


def test_daily_close_processes_three_missed_days_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    missed = [DAY - offset * date.resolution for offset in (3, 2, 1)]
    manifest = _manifest(closed=False)
    for candidate in missed:
        manifest["daily_verdicts"][candidate.isoformat()] = {  # type: ignore[index]
            "closure_ok": True
        }
    monkeypatch.setattr(coordinator, "_ready_manifest", lambda *_args: manifest)
    exported: list[date] = []
    monkeypatch.setattr(
        coordinator,
        "_daily_export",
        lambda _config, _root, candidate, *, sealed_only, **_kwargs: (
            exported.append(candidate)
            or {
                "package_status": "FINAL_CLOSED",
                "closure_ok": True,
                "reused": False,
            }
        ),
    )

    result = coordinator.run(config_path, "daily-close", day=DAY)

    assert result["status"] == "pending_closure"
    assert exported == missed


def test_daily_close_cannot_hide_failed_catch_up_behind_closed_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    older = DAY - date.resolution
    manifest = _manifest(closed=True)
    manifest["daily_verdicts"][older.isoformat()] = {  # type: ignore[index]
        "closure_ok": True
    }
    monkeypatch.setattr(coordinator, "_ready_manifest", lambda *_args: manifest)

    def export(
        _config: object,
        _root: Path,
        candidate: date,
        *,
        sealed_only: bool,
        expected_ready_manifest_sha256: str | None = None,
    ) -> dict[str, object]:
        assert sealed_only is True
        assert expected_ready_manifest_sha256 is None
        if candidate == older:
            raise RuntimeError("synthetic old-day export failure")
        return {
            "package_status": "FINAL_CLOSED",
            "closure_ok": True,
            "reused": False,
        }

    monkeypatch.setattr(coordinator, "_daily_export", export)

    result = coordinator.run(config_path, "daily-close", day=DAY)
    assert result["status"] == "failed"
    assert result["target_status"] == "closed"
    assert [item["status"] for item in result["attempts"]] == ["failed", "closed"]

    rc = coordinator.main(
        ["--config", str(config_path), "daily-close", "--day", DAY.isoformat()]
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["status"] == "failed"
    assert payload["target_status"] == "closed"


def test_standalone_daily_close_recovers_interrupted_ready_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ready = Path(config["pipeline_root"]) / "drop" / "mango_calls_ready.sqlite"
    ready.parent.mkdir(parents=True)
    ready.write_bytes(b"old-ready-generation")
    ready.chmod(0o600)
    old_manifest = {
        "ready_db": str(ready),
        "sha256": coordinator.sha256_file(ready),
        "size_bytes": ready.stat().st_size,
    }
    manifest_path = ready.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    staged = ready.parent / "staged.sqlite"
    staged.write_bytes(b"new-ready-generation")
    staged.chmod(0o600)
    new_manifest = {
        "ready_db": str(ready),
        "sha256": coordinator.sha256_file(staged),
        "size_bytes": staged.stat().st_size,
    }

    def crash(stage: str) -> None:
        if stage == "db_replaced":
            raise RuntimeError("synthetic publication interruption")

    with pytest.raises(RuntimeError, match="synthetic publication interruption"):
        commit_ready_generation(ready, staged, new_manifest, checkpoint=crash)
    assert inspect_ready_publication(ready)["recovery_required"] is True

    def recovered_manifest(*_args: object) -> dict[str, object]:
        assert inspect_ready_publication(ready)["recovery_required"] is False
        return _manifest(closed=False)

    monkeypatch.setattr(coordinator, "_ready_manifest", recovered_manifest)
    result = coordinator.run(config_path, "daily-close", day=DAY)

    assert result["status"] == "pending_closure"
    assert inspect_ready_publication(ready)["recovery_required"] is False
    assert ready.read_bytes() == b"new-ready-generation"


def test_daily_status_binds_decision_and_export_to_same_ready_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    expected_manifest_sha = "a" * 64
    monkeypatch.setattr(
        coordinator,
        "_ready_snapshot",
        lambda *_args: (_manifest(closed=False), expected_manifest_sha),
    )
    observed: list[str | None] = []

    def export(
        *_args: object,
        sealed_only: bool,
        expected_ready_manifest_sha256: str | None = None,
    ) -> dict[str, object]:
        assert sealed_only is False
        observed.append(expected_ready_manifest_sha256)
        return {
            "package_status": "INCOMPLETE_DO_NOT_USE_AS_FINAL",
            "reused": False,
        }

    monkeypatch.setattr(coordinator, "_daily_export", export)
    result = coordinator.run(config_path, "daily-status", day=DAY)

    assert result["status"] == "incomplete"
    assert observed == [expected_manifest_sha]


def test_daily_status_and_alert_exist_when_ready_manifest_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path, monkeypatch)

    status = coordinator.run(config_path, "daily-status", day=DAY)
    alert = coordinator.run(config_path, "daily-alert", day=DAY)

    assert status["status"] == "failed"
    assert status["error_type"] in {"FileNotFoundError", "RuntimeError"}
    assert Path(str(status["status_file"])).is_file()
    assert alert["status"] == "alert"
    assert "ready_manifest_unavailable" in alert["stop_reason"]
    assert Path(str(alert["alert"])).is_file()


def test_cli_normalizes_exception_to_safe_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        coordinator,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("secret detail must not escape")
        ),
    )

    rc = coordinator.main(
        ["--config", str(config_path), "current-plan", "--day", DAY.isoformat()]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 2
    assert payload["status"] == "failed"
    assert payload["stop_reason"] == "coordinator_exception:RuntimeError"
    assert "secret detail" not in json.dumps(payload)


def test_ready_manifest_binds_exact_code_host_and_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = tmp_path / "ready.sqlite"
    ready.write_bytes(b"sealed synthetic SQLite")
    host = tmp_path / "host_id"
    host.write_text("m1-host\n", encoding="utf-8")
    host.chmod(0o600)
    manifest = {
        "sha256": coordinator.sha256_file(ready),
        "size_bytes": ready.stat().st_size,
    }
    observed: list[tuple[object, object]] = []
    monkeypatch.setattr(
        coordinator,
        "stable_json_object",
        lambda *_args, **_kwargs: manifest,
    )

    def validate(_manifest: object, **kwargs: object) -> list[str]:
        observed.append(
            (kwargs.get("expected_code_sha"), kwargs.get("expected_host_id"))
        )
        return []

    monkeypatch.setattr(coordinator, "validate_ready_manifest_payload", validate)
    config = {"expected_code_sha": "a" * 40, "host_id_path": str(host)}

    assert coordinator._ready_manifest(config, ready) == manifest
    assert observed == [("a" * 40, "m1-host")]

    ready.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="does not match"):
        coordinator._ready_manifest(config, ready)


def test_config_and_publication_root_are_owner_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    config_path.chmod(0o644)
    with pytest.raises(RuntimeError, match="0600"):
        coordinator.run(config_path, "daily-status", day=DAY)

    config_path.chmod(0o600)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["publication_root"] = str(tmp_path / "outside-owner-local")
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="HOME/.mango_local"):
        coordinator.run(config_path, "daily-status", day=DAY)


def test_config_and_lock_symlinks_are_rejected_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _config(tmp_path, monkeypatch)
    config_link = tmp_path / "config-link.json"
    config_link.symlink_to(config_path)
    with pytest.raises(RuntimeError, match="config"):
        coordinator.run(config_link, "daily-status", day=DAY)

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(str(payload["publication_root"]))
    locks = root / "locks"
    locks.mkdir(parents=True, mode=0o700)
    victim = tmp_path / "lock-victim.txt"
    victim.write_text("must remain unchanged", encoding="utf-8")
    (locks / "coordinator.lock").symlink_to(victim)

    with pytest.raises(RuntimeError, match="lock is unsafe"):
        coordinator.run(config_path, "daily-status", day=DAY)
    assert victim.read_text(encoding="utf-8") == "must remain unchanged"
