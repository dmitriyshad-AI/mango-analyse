from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.productization.mango_calls_service_contract import (
    EXTERNAL_WATCHDOG_SCHEMA,
    load_owner_only_json,
    validate_external_watchdog_observation,
)


NOW = datetime(2026, 8, 11, 9, tzinfo=timezone.utc)
CODE_SHA = "a" * 40
CUTOVER_SHA = "b" * 64
SNAPSHOT_SHA = "c" * 64
PREVIOUS_HOST_ID = "source-mac"


def observation() -> dict[str, object]:
    return {
        "schema_version": EXTERNAL_WATCHDOG_SCHEMA,
        "observer_id": "external-observer-1",
        "observed_at_utc": "2026-08-11T08:59:00+00:00",
        "expected_code_sha": CODE_SHA,
        "cutover_manifest_sha256": CUTOVER_SHA,
        "previous_host": {
            "probe_ok": True,
            "host_id": PREVIOUS_HOST_ID,
            "shutdown_snapshot_sha256": SNAPSHOT_SHA,
            "active_calls_labels": [],
            "active_calls_pids": [],
        },
        "m1": {
            "probe_ok": True,
            "host_id": "m1-host",
            "heartbeat_at": "2026-08-11T08:58:00+00:00",
        },
    }


def validate(payload: object) -> dict[str, object]:
    return dict(
        validate_external_watchdog_observation(
            payload,
            expected_active_host_id="m1-host",
            expected_previous_host_id=PREVIOUS_HOST_ID,
            expected_code_sha=CODE_SHA,
            expected_cutover_manifest_sha256=CUTOVER_SHA,
            expected_previous_host_snapshot_sha256=SNAPSHOT_SHA,
            now=NOW,
        )
    )


def cli_command(source: Path) -> list[str]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check_mango_calls_external_watchdog.py"
    )
    return [
        sys.executable,
        str(script),
        "--observation",
        str(source),
        "--expected-active-host-id",
        "m1-host",
        "--expected-previous-host-id",
        PREVIOUS_HOST_ID,
        "--expected-code-sha",
        CODE_SHA,
        "--expected-cutover-manifest-sha256",
        CUTOVER_SHA,
        "--expected-previous-host-snapshot-sha256",
        SNAPSHOT_SHA,
    ]


def test_external_watchdog_green_observation_is_safe_and_read_only() -> None:
    report = validate(observation())

    assert report["status"] == "ok"
    assert report["ok"] is True
    assert report["errors"] == []
    assert report["safety"] == {
        "read_only_observation": True,
        "runs_asr": False,
        "runs_resolve_analyze": False,
        "writes_external_systems": False,
    }


def test_returned_old_host_label_or_pid_is_p0_without_raw_process_data() -> None:
    payload = observation()
    payload["previous_host"] = {
        "probe_ok": True,
        "host_id": PREVIOUS_HOST_ID,
        "shutdown_snapshot_sha256": SNAPSHOT_SHA,
        "active_calls_labels": ["com.mango.calls-pipeline"],
        "active_calls_pids": [4242],
    }

    report = validate(payload)
    serialized = json.dumps(report, ensure_ascii=False)

    assert report["status"] == "p0"
    assert report["ok"] is False
    assert "previous_host_calls_process_active" in report["errors"]
    assert report["previous_host_active_labels_count"] == 1
    assert report["previous_host_active_pids_count"] == 1
    assert "com.mango.calls-pipeline" not in serialized
    assert "4242" not in serialized


def test_unreachable_old_host_or_stale_m1_heartbeat_is_fail_closed() -> None:
    payload = observation()
    payload["previous_host"] = {
        "probe_ok": False,
        "host_id": PREVIOUS_HOST_ID,
        "shutdown_snapshot_sha256": SNAPSHOT_SHA,
        "active_calls_labels": [],
        "active_calls_pids": [],
    }
    payload["m1"] = {
        "probe_ok": True,
        "host_id": "m1-host",
        "heartbeat_at": "2026-08-11T08:00:00+00:00",
    }

    report = validate(payload)

    assert report["status"] == "alert"
    assert "previous_host_probe_unproven" in report["errors"]
    assert "m1_heartbeat_stale_or_future" in report["errors"]


def test_invalid_observer_id_is_not_reflected_into_safe_report() -> None:
    payload = observation()
    payload["observer_id"] = "dmitriy@example.com"

    report = validate(payload)
    serialized = json.dumps(report, ensure_ascii=False)

    assert report["status"] == "alert"
    assert report["observer_id_valid"] is False
    assert "observer_id_invalid" in report["errors"]
    assert "dmitriy@example.com" not in serialized


def test_external_watchdog_is_bound_to_previous_host_and_shutdown_snapshot() -> None:
    payload = observation()
    payload["previous_host"] = {
        "probe_ok": True,
        "host_id": "wrong-source-mac",
        "shutdown_snapshot_sha256": "d" * 64,
        "active_calls_labels": [],
        "active_calls_pids": [],
    }

    report = validate(payload)

    assert report["status"] == "alert"
    assert "previous_host_id_mismatch" in report["errors"]
    assert "previous_host_snapshot_sha256_mismatch" in report["errors"]


def test_external_watchdog_cli_requires_owner_only_bound_observation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "observation.json"
    payload = observation()
    current = datetime.now(timezone.utc)
    payload["observed_at_utc"] = current.isoformat()
    assert isinstance(payload["m1"], dict)
    payload["m1"]["heartbeat_at"] = current.isoformat()
    source.write_text(json.dumps(payload), encoding="utf-8")
    source.chmod(0o600)
    command = cli_command(source)

    accepted = subprocess.run(command, text=True, capture_output=True)
    assert accepted.returncode == 0
    assert json.loads(accepted.stdout)["status"] == "ok"

    source.chmod(0o644)
    rejected = subprocess.run(command, text=True, capture_output=True)
    assert rejected.returncode == 2
    payload = json.loads(rejected.stdout)
    assert payload["status"] == "alert"
    assert payload["errors"] == ["watchdog_input_error:RuntimeError"]


def test_external_watchdog_cli_has_no_clock_override(tmp_path: Path) -> None:
    source = tmp_path / "observation.json"
    payload = observation()
    payload["observed_at_utc"] = "2000-01-01T00:00:00+00:00"
    assert isinstance(payload["m1"], dict)
    payload["m1"]["heartbeat_at"] = "2000-01-01T00:00:00+00:00"
    source.write_text(json.dumps(payload), encoding="utf-8")
    source.chmod(0o600)

    stale = subprocess.run(cli_command(source), text=True, capture_output=True)
    assert stale.returncode == 2
    assert "observation_stale_or_future" in json.loads(stale.stdout)["errors"]

    overridden = subprocess.run(
        [*cli_command(source), "--now", "2000-01-01T00:00:00+00:00"],
        text=True,
        capture_output=True,
    )
    assert overridden.returncode == 2
    assert "unrecognized arguments: --now" in overridden.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS extended ACL syntax")
def test_owner_only_observation_rejects_extended_acl(tmp_path: Path) -> None:
    source = tmp_path / "observation.json"
    payload = observation()
    current = datetime.now(timezone.utc)
    payload["observed_at_utc"] = current.isoformat()
    assert isinstance(payload["m1"], dict)
    payload["m1"]["heartbeat_at"] = current.isoformat()
    source.write_text(json.dumps(payload), encoding="utf-8")
    source.chmod(0o600)
    subprocess.run(
        ["/bin/chmod", "+a", "everyone allow read", str(source)],
        check=True,
    )
    try:
        with pytest.raises(RuntimeError, match="must_not_have_extended_acl"):
            load_owner_only_json(source, label="external_watchdog_observation")
        rejected = subprocess.run(
            cli_command(source), text=True, capture_output=True
        )
    finally:
        subprocess.run(["/bin/chmod", "-N", str(source)], check=True)

    assert rejected.returncode == 2
    assert json.loads(rejected.stdout)["errors"] == [
        "watchdog_input_error:RuntimeError"
    ]
