from __future__ import annotations

import json
import importlib.util
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts/bootstrap_m1_mango_calls.sh"
HANDOFF = ROOT / "docs/m1_calls_handoff_20260801"
ENV_PARSER = ROOT / "scripts/mango_calls_env.py"
CONTROLLED_ALLOWLIST = ROOT / "scripts/create_m1_calls_controlled_allowlist.py"


def _load_env_parser():
    spec = importlib.util.spec_from_file_location("mango_calls_env", ENV_PARSER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_controlled_allowlist_script():
    spec = importlib.util.spec_from_file_location(
        "create_m1_calls_controlled_allowlist",
        CONTROLLED_ALLOWLIST,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controlled_allowlist_verifies_lineage_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controlled_allowlist_script()
    calls: list[bool] = []

    def failed_authority(_config):
        calls.append(True)
        return {"ok": False, "controlled_cursor_binding_ok": False}

    monkeypatch.setattr(
        module,
        "controlled_read_only_cutover_authority_report",
        failed_authority,
    )
    with pytest.raises(
        RuntimeError,
        match="controlled_allowlist_cutover_lineage_unproven",
    ):
        module.verify_controlled_cutover_lineage(object())
    assert calls == [True]


def test_controlled_allowlist_accepts_only_proven_read_only_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_controlled_allowlist_script()

    monkeypatch.setattr(
        module,
        "controlled_read_only_cutover_authority_report",
        lambda _config: {
            "ok": True,
            "controlled_cursor_binding_ok": True,
        },
    )

    assert module.verify_controlled_cutover_lineage(object()) is None
    source = CONTROLLED_ALLOWLIST.read_text(encoding="utf-8")
    lease_call = source.index("with process_a_leases(")
    snapshot_call = source.index("db_snapshot = controlled_call_database_snapshot(")
    lineage_call = source.index("verify_controlled_cutover_lineage(config)")
    allowlist_write = source.index("atomic_replace_owner_only_bytes(\n                out,")
    allowlist_readback = source.index("loaded = load_controlled_call_allowlist(")
    assert (
        lease_call
        < snapshot_call
        < lineage_call
        < allowlist_write
        < allowlist_readback
    )


def test_bootstrap_plan_is_safe_and_complete() -> None:
    result = subprocess.run(
        [str(BOOTSTRAP), "plan"], cwd=ROOT, text=True, capture_output=True, check=True
    )
    payload = json.loads(result.stdout)

    assert payload["starts_services"] is False
    assert payload["runs_asr"] is False
    assert payload["runs_resolve_analyze"] is False
    assert payload["writes_external_systems"] is False
    assert {"git", "ffmpeg", "node", "python@3.12"} <= set(payload["system_packages"])
    assert "Tallanto contacts CSV" in payload["report_access"]
    assert "requirements-local-whisper.txt" in payload["python_requirements"]
    assert "requirements-local-dual-asr.txt" in payload["python_requirements"]
    assert "private Google Drive folder" in payload["optional_publish_access"]
    assert (ROOT / "requirements-local-whisper.txt").read_text(encoding="utf-8").splitlines() == [
        "mlx-whisper==0.4.3",
        "imageio-ffmpeg==0.6.0",
    ]
    assert (ROOT / "requirements-local-dual-asr.txt").read_text(encoding="utf-8").strip() == "gigaam==0.1.0"


def test_package_install_requires_exact_confirmation() -> None:
    environment = {key: value for key, value in os.environ.items() if key != "CONFIRM_M1_PACKAGE_INSTALL"}
    result = subprocess.run(
        [str(BOOTSTRAP), "install"], cwd=ROOT, env=environment, text=True, capture_output=True
    )

    assert result.returncode == 3
    assert "INSTALL_M1_MANGO_CALLS_PACKAGES" in result.stderr
    assert "install_packages() {\n  umask 077\n" in BOOTSTRAP.read_text(encoding="utf-8")


def test_bootstrap_cannot_start_pipeline_or_launchd() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")

    assert "/bin/launchctl print" in source
    assert "launchctl bootstrap" not in source
    assert "launchctl bootout" not in source
    assert "run_mango_calls_pipeline.py" not in source
    assert "mango_mvp.cli worker" not in source
    assert "--execute" not in source
    assert "@openai/codex@0.142.3" in source
    assert '"tallanto_export_owner_only":%s' in source
    assert '"google_publish_enabled":%s' in source
    assert '"google_config_valid":%s' in source
    assert '"yandex_target_verified":%s' in source
    assert '"disk_space_ok":%s' in source
    assert '"network_access_verified":false' in source
    assert "/usr/bin/env -i" in source
    assert "ls-files -v" in source
    assert "diff-files --quiet --ignore-submodules=none" in source
    assert "diff-index --cached --quiet --ignore-submodules=none" in source
    assert '&& "$tallanto_export" == true' in source
    assert '&& "$google_config_valid" == true' in source
    assert '&& "$google_config_valid" == true && "$secrets_dir_owner_only"' in source
    assert '&& "$yandex" == true && "$secrets_dir_owner_only"' not in source
    assert '"external_publication_ready":%s' in source
    assert '"developer_profile_ready":%s' in source
    assert '&& "$skills" == true' not in source
    assert '&& "$owner_local_root_only" == true' in source
    assert '&& "$pipeline_root_owner_only" == true' in source
    assert '&& "$pipeline_root_under_owner_local" == true' in source
    assert '&& "$pipeline_root_matches_env" == true' in source
    assert '"pipeline_root_under_owner_local":%s' in source
    assert '"owner_local_root_only":%s' in source
    assert '"pipeline_root_matches_env":%s' in source
    assert '"host_preflight_passed":%s' in source
    assert '"runtime_ready":false' in source
    assert '&& "$disk_space_ok" == true' in source
    assert '&& "$conflicting_services_loaded" == false' in source
    assert '&& "$pipeline_lock_held" == false' in source
    assert '${GOOGLE_APPLICATION_CREDENTIALS:-' not in source
    assert 'manifest["local_skills"]' in source
    assert '== "codex-cli 0.142.3"' in source
    for label in (
        "com.mango.calls-capture",
        "com.mango.calls-pipeline",
        "com.mango.calls-watchdog",
        "com.mango.calls-publication-close-0600",
        "com.mango.calls-publication-close-0700",
        "com.mango.calls-publication-close-0800",
        "com.mango.calls-publication-alert-0830",
        "com.mango.calls-publication-status-0850",
    ):
        assert label in source
    assert '"capture.lock", "pipeline.lock"' in source

    config = json.loads((HANDOFF / "config.m1.example.json").read_text(encoding="utf-8"))
    assert "Projects/Mango analyse" not in config["pipeline_root"]
    assert config["pipeline_root"] == "<HOME>/.mango_local/mango_calls_two_processes"


def test_google_is_optional_but_partial_google_config_is_blocked() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")

    assert '[[ -z "$google_path" && -z "$google_folder_id" ]]' in source
    assert '[[ "$google" == true ]] && google_config_valid=true' in source
    env_example = (HANDOFF / "mango_calls_m1_worker.env.example").read_text(encoding="utf-8")
    assert "MANGO_CALLS_GOOGLE_DRIVE_FOLDER_ID" not in env_example
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env_example


def test_handoff_examples_contain_no_secret_values() -> None:
    env_lines = (HANDOFF / "mango_calls_m1_worker.env.example").read_text(encoding="utf-8").splitlines()
    secret_keys = {
        "MANGO_OFFICE_API_KEY",
        "MANGO_OFFICE_API_SALT",
    }
    values = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in env_lines
        if line and not line.startswith("#") and "=" in line
    }

    assert all(values[key] == "" for key in secret_keys)
    json.loads((HANDOFF / "config.m1.example.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (HANDOFF / "codex_profile_manifest_20260801.json").read_text(encoding="utf-8")
    )
    assert manifest["contains_secrets"] is False
    assert manifest["schema_version"] == "codex_profile_manifest_v2"
    assert manifest["transfer_archive"] is None
    assert manifest["transfer_archive_status"] == "must_be_regenerated_after_cutover_sha_is_frozen"
    assert "mango-progressive-data-rollout" in manifest["local_skills"]
    assert "doc" not in manifest["local_skills"]
    assert "imagegen" in manifest["bundled_skills_reinstalled_with_codex"]
    assert "imagegen" not in manifest["local_skills"]
    assert {item["name"] for item in manifest["mcp_and_connectors"]} >= {
        "github",
        "node_repl",
        "Todoist",
    }


def test_prompt_forbids_live_start_and_secret_transport() -> None:
    prompt = (HANDOFF / "M1_CODEX_PROMPT.md").read_text(encoding="utf-8")

    assert "Не\n   запускай ASR, Resolve, Analyze" in prompt
    assert "--install и запуск службы выполняй только после отдельного подтверждения" in prompt
    assert "auth.json не копируй" in prompt
    assert "production_ready=false" in prompt


def test_check_with_empty_home_returns_json_and_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as home:
        result = subprocess.run(
            [str(BOOTSTRAP), "check"], cwd=ROOT, env={**os.environ, "HOME": home},
            text=True, capture_output=True,
        )

    payload = json.loads(result.stdout)
    assert result.returncode != 0
    assert payload["network_access_verified"] is False
    assert payload["config_owner_only_and_valid"] is False
    assert payload["clean_expected_revision"] is False
    assert result.stderr == ""


def test_owner_local_root_mode_and_symlink_are_reported_fail_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    local = home / ".mango_local"
    local.mkdir(mode=0o755)
    local.chmod(0o755)

    def check() -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        result = subprocess.run(
            [str(BOOTSTRAP), "check"],
            cwd=ROOT,
            env={**os.environ, "HOME": str(home)},
            text=True,
            capture_output=True,
        )
        return result, json.loads(result.stdout)

    result, payload = check()
    assert result.returncode != 0
    assert payload["owner_local_root_only"] is False

    local.chmod(0o700)
    result, payload = check()
    assert result.returncode != 0
    assert payload["owner_local_root_only"] is True

    local.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    local.symlink_to(outside, target_is_directory=True)
    result, payload = check()
    assert result.returncode != 0
    assert payload["owner_local_root_only"] is False


def test_secrets_root_symlink_is_reported_fail_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    outside = tmp_path / "outside-secrets"
    outside.mkdir(mode=0o700)
    (home / ".mango_secrets").symlink_to(outside, target_is_directory=True)

    result = subprocess.run(
        [str(BOOTSTRAP), "check"],
        cwd=ROOT,
        env={**os.environ, "HOME": str(home)},
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert json.loads(result.stdout)["secrets_dir_owner_only"] is False


def test_readiness_probe_accepts_tuple_and_rejects_future_measurement(
    tmp_path: Path, monkeypatch
) -> None:
    from mango_mvp.productization.mango_office_client import MangoOfficeClient
    from scripts import probe_m1_calls_access as probe

    monkeypatch.setattr(
        probe,
        "_owner_env",
        lambda _path: {
            "MANGO_OFFICE_API_KEY": "synthetic-key",
            "MANGO_OFFICE_API_SALT": "synthetic-salt",
            "MANGO_OFFICE_BASE_URL": "https://example.invalid",
        },
    )
    monkeypatch.setattr(
        MangoOfficeClient,
        "poll_call_history",
        lambda _self, **_kwargs: (),
    )
    assert probe.probe_mango_readonly({"mango_env_path": str(tmp_path / "unused")})

    future = datetime.now(timezone.utc) + timedelta(minutes=1)
    assert not probe._measurement_evidence_ok(
        {
            "schema_version": "m1_mango_calls_measurements_v1",
            "expected_code_sha": "a" * 40,
            "host_id": "m1-host",
            "captured_at_utc": future.isoformat(),
        },
        expected_sha="a" * 40,
        host_id="m1-host",
    )


def test_codex_model_probe_uses_exact_models_and_isolated_wrapper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import probe_m1_calls_access as probe

    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    config = SimpleNamespace(
        codex_binary=codex,
        codex_home_root=tmp_path / "codex-homes",
        strict_ready_provenance=True,
        codex_resolve_model="resolve-exact",
        codex_analyze_model="analyze-exact",
        codex_reasoning_effort="medium",
        codex_service_tier="flex",
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def completed(command, **kwargs):
        if "--output-last-message" in command:
            Path(command[command.index("--output-last-message") + 1]).write_text(
                "OK\n",
                encoding="utf-8",
            )
        calls.append(([str(item) for item in command], dict(kwargs)))
        return SimpleNamespace(returncode=0)

    for name in (
        "MANGO_OFFICE_API_SALT",
        "TALLANTO_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "YANDEX_DISK_TOKEN",
        "OPENAI_API_KEY",
    ):
        monkeypatch.setenv(name, f"sentinel-{name}")
    monkeypatch.setattr(probe, "codex_network_available", lambda: True)
    monkeypatch.setattr(
        probe,
        "ensure_codex_runtime_anchor",
        lambda _config: tmp_path,
    )
    monkeypatch.setattr(
        probe,
        "prepare_codex_home",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(probe.subprocess, "run", completed)

    first = probe.probe_codex_models(config)
    second = probe.probe_codex_models(config)

    assert all(first.values())
    assert all(second.values())
    model_commands = [command for command, _kwargs in calls if "--model" in command]
    assert len(model_commands) == 4
    assert [
        command[command.index("--model") + 1] for command in model_commands
    ] == ["resolve-exact", "analyze-exact"] * 2
    assert all("run_codex_cli_isolated.sh" in command[0] for command in model_commands)
    assert all("--ephemeral" not in command for command in model_commands)
    assert all("--output-last-message" in command for command in model_commands)
    assert all('service_tier="flex"' in command for command in model_commands)
    first_model_calls = [
        (command, kwargs)
        for command, kwargs in calls
        if "--model" in command
    ][:2]
    resolve_command, resolve_kwargs = first_model_calls[0]
    analyze_command, analyze_kwargs = first_model_calls[1]
    assert "--ignore-user-config" not in resolve_command
    assert resolve_command[-1].endswith("Reply with exactly OK.")
    assert resolve_kwargs["input"] is None
    assert "--ignore-user-config" in analyze_command
    assert analyze_command[-1] == "-"
    assert analyze_kwargs["input"].endswith("Reply with exactly OK.")
    expected_env_keys = {
        "HOME",
        "CODEX_HOME",
        "PATH",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "NO_COLOR",
        "MANGO_CODEX_REAL_BIN",
        "MANGO_CODEX_PROCESS_HOME",
        "MANGO_CODEX_PROCESS_TMPDIR",
    }
    for command, kwargs in calls:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert set(environment) <= expected_env_keys
        assert not {
            "MANGO_OFFICE_API_SALT",
            "TALLANTO_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "YANDEX_DISK_TOKEN",
            "OPENAI_API_KEY",
        } & set(environment)
        assert kwargs["cwd"] == Path(str(environment["HOME"]))
        if "--model" in command:
            assert set(environment) == expected_env_keys
            assert environment["MANGO_CODEX_REAL_BIN"] == str(codex)
            assert environment["MANGO_CODEX_PROCESS_HOME"] == environment["HOME"]
            assert (
                environment["MANGO_CODEX_PROCESS_TMPDIR"]
                == environment["TMPDIR"]
            )
        else:
            assert set(environment) == expected_env_keys - {
                "MANGO_CODEX_REAL_BIN",
                "MANGO_CODEX_PROCESS_HOME",
                "MANGO_CODEX_PROCESS_TMPDIR",
            }
            assert "MANGO_CODEX_REAL_BIN" not in environment
    assert not list(tmp_path.glob(".mango-codex-model-probe-*"))
    assert not list(config.codex_home_root.glob("probe-*"))


def test_codex_model_probe_fails_closed_on_auth_network_or_model_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import probe_m1_calls_access as probe

    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    config = SimpleNamespace(
        codex_binary=codex,
        codex_home_root=tmp_path / "codex-homes",
        strict_ready_provenance=True,
        codex_resolve_model="resolve-exact",
        codex_analyze_model="analyze-exact",
        codex_reasoning_effort="medium",
        codex_service_tier="flex",
    )
    monkeypatch.setattr(
        probe,
        "prepare_codex_home",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        probe,
        "ensure_codex_runtime_anchor",
        lambda _config: tmp_path,
    )

    monkeypatch.setattr(probe, "codex_network_available", lambda: False)
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    no_network = probe.probe_codex_models(config)
    assert no_network == {
        "codex_auth_probe_attempted": True,
        "codex_resolve_model_access_attempted": False,
        "codex_analyze_model_access_attempted": False,
        "codex_resolve_model_access_completed": False,
        "codex_analyze_model_access_completed": False,
        "codex_authenticated_ok": True,
        "codex_network_ok": False,
        "codex_resolve_model_access_ok": False,
        "codex_analyze_model_access_ok": False,
    }

    monkeypatch.setattr(probe, "codex_network_available", lambda: True)

    def auth_failure(command, **_kwargs):
        return SimpleNamespace(returncode=1 if "login" in command else 0)

    monkeypatch.setattr(probe.subprocess, "run", auth_failure)
    auth_failed = probe.probe_codex_models(config)
    assert auth_failed["codex_authenticated_ok"] is False
    assert auth_failed["codex_resolve_model_access_ok"] is False
    assert auth_failed["codex_analyze_model_access_ok"] is False

    def analyze_failure(command, **_kwargs):
        if "--output-last-message" in command:
            model = command[command.index("--model") + 1]
            Path(command[command.index("--output-last-message") + 1]).write_text(
                "NOT_OK\n" if model == "analyze-exact" else "OK\n",
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(probe.subprocess, "run", analyze_failure)
    analyze_failed = probe.probe_codex_models(config)
    assert analyze_failed["codex_resolve_model_access_ok"] is True
    assert analyze_failed["codex_analyze_model_access_ok"] is False


def test_readiness_probe_requires_europe_moscow_timezone(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import probe_m1_calls_access as probe

    host_id_file = tmp_path / "host_id"
    host_id_file.write_text("m1-host\n", encoding="utf-8")
    host_id_file.chmod(0o600)
    parsed_config = SimpleNamespace(
        host_id_file=host_id_file,
        strict_ready_provenance=True,
        expected_active_host_id="m1-host",
        expected_previous_host_id="source-mac",
        processing_scope="controlled_1",
        stage_limit=1,
        working_db=tmp_path / "working.sqlite",
        working_audio_dir=tmp_path / "working" / "audio",
    )
    config = {
        "pipeline_root": str(tmp_path / "pipeline"),
        "codex_home_root": str(tmp_path / "codex"),
        "expected_code_sha": "a" * 40,
        "expected_active_host_id": "m1-host",
    }
    approved = probe.approved_runtime_fingerprint()
    monkeypatch.setattr(probe, "current_git_sha", lambda _root: "a" * 40)
    monkeypatch.setattr(probe, "git_worktree_is_clean", lambda _root: True)
    monkeypatch.setattr(
        probe.pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(pw_name="dmitriy"),
    )
    monkeypatch.setattr(
        probe,
        "cutover_authority_report",
        lambda *_args, **_kwargs: {
            "ok": True,
            "active_host_id": "m1-host",
        },
    )
    monkeypatch.setattr(
        probe,
        "controlled_read_only_cutover_authority_report",
        lambda *_args, **_kwargs: {
            "ok": True,
            "active_host_id": "m1-host",
            "controlled_cursor_binding_ok": True,
        },
    )
    monkeypatch.setattr(
        probe,
        "controlled_call_scope_for_config",
        lambda _config: SimpleNamespace(
            source_call_id="TARGET",
            target_record_id=2,
            source_audio_sha256="c" * 64,
            source_audio_size_bytes=100,
            allowlist_sha256="b" * 64,
            code_sha="a" * 40,
            tenant_id="foton",
            host_id="m1-host",
        ),
    )
    monkeypatch.setattr(
        probe,
        "controlled_call_bound_snapshot",
        lambda *_args, **_kwargs: {
            "target": {
                "record_id": 2,
                "ready_for_human_review": False,
                "source_audio": {"sha256": "c" * 64, "size_bytes": 100},
            }
        },
    )
    monkeypatch.setattr(probe, "command_output", lambda _command: "")
    monkeypatch.setattr(probe, "physical_memory_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(
        probe.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100 * 1024**3),
    )
    monkeypatch.setattr(probe.shutil, "which", lambda _name: "/synthetic/tool")
    monkeypatch.setattr(
        probe,
        "observe_runtime_fingerprint",
        lambda _config: {"ok": True, "fingerprint": approved, "errors": []},
    )
    monkeypatch.setattr(probe, "inspect_codex_home", lambda _path: {"ok": True})
    monkeypatch.setattr(
        probe,
        "probe_google_readonly",
        lambda _config: {
            "google_spreadsheet_acl_ok": True,
            "google_metadata_readback_ok": True,
        },
    )
    monkeypatch.setattr(probe, "probe_mango_readonly", lambda _config: True)
    monkeypatch.setattr(probe, "probe_tallanto_readonly", lambda _config: True)
    monkeypatch.setattr(probe, "probe_yandex_marker", lambda _config: True)
    monkeypatch.setattr(
        probe,
        "probe_offline_models",
        lambda _config: {
            "offline_whisper_synthetic_attempted": True,
            "offline_gigaam_synthetic_attempted": True,
            "offline_whisper_synthetic_completed": True,
            "offline_gigaam_synthetic_completed": True,
            "offline_whisper_synthetic_ok": True,
            "offline_gigaam_synthetic_ok": True,
        },
    )
    monkeypatch.setattr(
        probe,
        "probe_codex_models",
        lambda _config: {
            "codex_auth_probe_attempted": True,
            "codex_resolve_model_access_attempted": True,
            "codex_analyze_model_access_attempted": True,
            "codex_resolve_model_access_completed": True,
            "codex_analyze_model_access_completed": True,
            "codex_authenticated_ok": True,
            "codex_network_ok": True,
            "codex_resolve_model_access_ok": True,
            "codex_analyze_model_access_ok": True,
        },
    )
    monkeypatch.setattr(probe, "probe_time_sync", lambda: True)
    monkeypatch.setattr(probe, "probe_conflicting_launchd", lambda: True)
    monkeypatch.setattr(
        probe,
        "_measurement_evidence_ok",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        probe,
        "stage_capacity_report",
        lambda **_kwargs: {"status": "ok", "capacity_ok": True},
    )
    evidence = {"controlled_10": {}, "mango_peak_60d": {}}

    monkeypatch.setattr(probe, "machine_timezone", lambda: "Europe/Moscow")
    ready = probe.probe(
        config,
        evidence,
        parsed_config=parsed_config,
        run_offline_model_probes=True,
        run_codex_model_probes=True,
    )
    assert ready["checks"]["timezone_is_europe_moscow"] is True
    assert ready["host_readiness"] == "STOP"
    assert ready["controlled_1_readiness"]["status"] == "OK"
    assert ready["controlled_1_readiness"]["requires_controlled_10"] is False
    assert ready["service_capacity_readiness"]["status"] == "OK"
    assert ready["service_capacity_readiness"]["requires_controlled_10"] is True
    assert ready["service_machine_preflight"]["status"] == "STOP"
    assert ready["production_service_readiness"]["status"] == "STOP"
    assert (
        ready["mode"]
        == "read_only_access_plus_synthetic_offline_and_codex_models"
    )
    assert ready["machine"]["timezone"] == "Europe/Moscow"
    assert ready["checks"]["active_m1_host_identity_bound"] is True
    assert ready["checks"]["runtime_user_is_dmitriy"] is True
    assert ready["checks"]["controlled_one_allowlist_bound"] is True
    assert ready["checks"]["controlled_one_target_unique_in_working_db"] is True
    assert ready["checks"]["processing_scope_is_controlled_one"] is True
    assert ready["checks"]["processing_scope_is_service"] is False
    assert ready["runs_content_free_network_model_inference"] is True
    assert ready["runs_codex_model_access_probes"] is True
    assert ready["runs_asr"] is True
    assert ready["runs_resolve_analyze_pipeline"] is False
    assert ready["uses_customer_content"] is False

    parsed_config.processing_scope = "service"
    monkeypatch.setattr(
        probe,
        "controlled_call_scope_for_config",
        lambda _config: None,
    )
    service_ready = probe.probe(
        config,
        evidence,
        parsed_config=parsed_config,
        run_offline_model_probes=True,
        run_codex_model_probes=True,
    )
    assert service_ready["controlled_1_readiness"]["status"] == "STOP"
    assert service_ready["service_capacity_readiness"]["status"] == "OK"
    assert service_ready["service_machine_preflight"]["status"] == "OK"
    assert service_ready["production_service_readiness"]["status"] == "STOP"
    assert service_ready["production_service_readiness"]["requires_controlled_1"] is True
    assert service_ready["production_service_readiness"]["does_not_authorize_launch"] is True
    assert (
        service_ready["production_service_readiness"]["checks"]
        ["controlled_1_human_pass_verified"]
        is False
    )
    assert service_ready["host_readiness"] == "STOP"
    assert service_ready["checks"]["processing_scope_is_controlled_one"] is False
    assert service_ready["checks"]["processing_scope_is_service"] is True

    parsed_config.processing_scope = "controlled_1"

    monkeypatch.setattr(
        probe,
        "controlled_call_scope_for_config",
        lambda _config: None,
    )
    missing_allowlist = probe.probe(
        config,
        evidence,
        parsed_config=parsed_config,
        run_offline_model_probes=True,
        run_codex_model_probes=True,
    )
    assert missing_allowlist["controlled_1_readiness"]["status"] == "STOP"
    assert missing_allowlist["checks"]["controlled_one_allowlist_bound"] is False

    monkeypatch.setattr(
        probe,
        "controlled_call_scope_for_config",
        lambda _config: SimpleNamespace(
            source_call_id="TARGET",
            target_record_id=2,
            source_audio_sha256="c" * 64,
            source_audio_size_bytes=100,
            allowlist_sha256="b" * 64,
            code_sha="a" * 40,
            tenant_id="foton",
            host_id="m1-host",
        ),
    )

    offline_only = probe.probe(
        config,
        evidence,
        parsed_config=parsed_config,
        run_offline_model_probes=True,
    )
    assert (
        offline_only["mode"]
        == "read_only_access_plus_synthetic_offline_models"
    )
    assert offline_only["controlled_1_readiness"]["status"] == "STOP"

    monkeypatch.setattr(
        probe,
        "_measurement_evidence_ok",
        lambda *_args, **_kwargs: False,
    )
    without_controlled_10 = probe.probe(
        config,
        {},
        parsed_config=parsed_config,
        run_offline_model_probes=True,
        run_codex_model_probes=True,
    )
    assert without_controlled_10["controlled_1_readiness"]["status"] == "OK"
    assert without_controlled_10["service_capacity_readiness"]["status"] == "STOP"
    assert without_controlled_10["production_service_readiness"]["status"] == "STOP"
    assert without_controlled_10["host_readiness"] == "STOP"

    wrong_host_config = {**config, "expected_active_host_id": "another-m1"}
    wrong_host = probe.probe(
        wrong_host_config,
        evidence,
        parsed_config=parsed_config,
        run_offline_model_probes=True,
        run_codex_model_probes=True,
    )
    assert wrong_host["checks"]["active_m1_host_identity_bound"] is False
    assert wrong_host["controlled_1_readiness"]["status"] == "STOP"

    monkeypatch.setattr(
        probe,
        "_measurement_evidence_ok",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(probe, "machine_timezone", lambda: "UTC")
    wrong_timezone = probe.probe(
        config,
        evidence,
        parsed_config=parsed_config,
        run_offline_model_probes=True,
        run_codex_model_probes=True,
    )
    assert wrong_timezone["checks"]["timezone_is_europe_moscow"] is False
    assert wrong_timezone["host_readiness"] == "STOP"
    assert wrong_timezone["controlled_1_readiness"]["status"] == "STOP"
    assert wrong_timezone["service_capacity_readiness"]["status"] == "OK"
    assert wrong_timezone["production_service_readiness"]["status"] == "STOP"

    monkeypatch.setattr(probe, "machine_timezone", lambda: "Europe/Moscow")
    monkeypatch.setattr(
        probe,
        "probe_codex_models",
        lambda _config: {
            "codex_auth_probe_attempted": True,
            "codex_resolve_model_access_attempted": True,
            "codex_analyze_model_access_attempted": True,
            "codex_resolve_model_access_completed": True,
            "codex_analyze_model_access_completed": True,
            "codex_authenticated_ok": True,
            "codex_network_ok": True,
            "codex_resolve_model_access_ok": True,
            "codex_analyze_model_access_ok": False,
        },
    )
    unavailable_analyze = probe.probe(
        config,
        evidence,
        parsed_config=parsed_config,
        run_offline_model_probes=True,
        run_codex_model_probes=True,
    )
    assert unavailable_analyze["controlled_1_readiness"]["status"] == "STOP"
    assert unavailable_analyze["production_service_readiness"]["status"] == "STOP"

    monkeypatch.setattr(
        probe,
        "probe_codex_models",
        lambda _config: {
            "codex_auth_probe_attempted": True,
            "codex_resolve_model_access_attempted": True,
            "codex_analyze_model_access_attempted": True,
            "codex_resolve_model_access_completed": True,
            "codex_analyze_model_access_completed": True,
            "codex_authenticated_ok": True,
            "codex_network_ok": True,
            "codex_resolve_model_access_ok": False,
            "codex_analyze_model_access_ok": False,
        },
    )
    failed_model_requests = probe.probe(
        config,
        evidence,
        parsed_config=parsed_config,
        run_offline_model_probes=True,
        run_codex_model_probes=True,
    )
    assert failed_model_requests["codex_model_access_probe_attempted"] is True
    assert failed_model_requests["runs_codex_model_access_probes"] is True
    assert failed_model_requests["external_model_requests_made"] is None

    requested_without_parsed_config = probe.probe(
        config,
        evidence,
        parsed_config=None,
        run_offline_model_probes=True,
        run_codex_model_probes=True,
    )
    assert requested_without_parsed_config["offline_model_probes_requested"] is True
    assert requested_without_parsed_config["codex_model_access_probes_requested"] is True
    assert requested_without_parsed_config["runs_asr"] is False
    assert (
        requested_without_parsed_config["runs_codex_model_access_probes"]
        is False
    )
    assert (
        requested_without_parsed_config[
            "runs_content_free_network_model_inference"
        ]
        is False
    )


def test_readiness_probe_cli_selects_controlled_1_without_weakening_service(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import probe_m1_calls_access as probe

    monkeypatch.setattr(probe, "load_owner_only_json", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        probe.CallsTwoProcessesConfig,
        "from_json",
        lambda _path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        probe,
        "probe",
        lambda *_args, **_kwargs: {
            "host_readiness": "STOP",
            "controlled_1_readiness": {"status": "OK", "ok": True},
            "service_capacity_readiness": {"status": "STOP", "ok": False},
            "production_service_readiness": {"status": "STOP", "ok": False},
        },
    )
    config_path = tmp_path / "config.json"

    assert probe.main(["--config", str(config_path)]) == 1
    assert probe.main(
        [
            "--config",
            str(config_path),
            "--readiness-target",
            "controlled-1",
        ]
    ) == 0


def test_ephemeral_codex_cycle_is_checked_for_persistent_residue(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import probe_m1_calls_access as probe

    home = tmp_path / "home"
    runtime_home = home / ".mango_local" / "codex-runtime"
    runtime_home.mkdir(parents=True, mode=0o700)
    runtime_home.chmod(0o700)
    auth = runtime_home / "auth.json"
    auth.write_text('{"synthetic":true}', encoding="utf-8")
    auth.chmod(0o600)
    captured = tmp_path / "args.txt"
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/bin/zsh\n"
        f"printf '%s\\n' \"$@\" > {shlex.quote(str(captured))}\n",
        encoding="utf-8",
    )
    fake.chmod(0o700)
    wrapper = ROOT / "scripts" / "run_codex_cli_isolated.sh"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    process_tmp = tmp_path / "process-tmp"
    process_tmp.mkdir(mode=0o700)
    env = {
        **os.environ,
        "HOME": str(home),
        "CODEX_HOME": str(runtime_home),
        "MANGO_CODEX_REAL_BIN": str(fake),
        "MANGO_CODEX_PROCESS_HOME": str(home),
        "MANGO_CODEX_PROCESS_TMPDIR": str(process_tmp),
    }

    subprocess.run(
        [str(wrapper), "exec", "--model", "synthetic", "safe synthetic prompt"],
        env=env,
        check=True,
    )

    args = captured.read_text(encoding="utf-8").splitlines()
    assert "--ephemeral" in args
    assert probe.inspect_codex_home(runtime_home)["ok"] is True

    for label in (
        "worker",
        "transcribe",
        "backfill_second_asr",
        "resolve",
        "analyze",
    ):
        stage_home = runtime_home / label
        stage_home.mkdir(mode=0o700)
        stage_home.chmod(0o700)
        stage_config = stage_home / "config.toml"
        stage_config.write_text("# synthetic\n", encoding="utf-8")
        stage_config.chmod(0o600)
    assert probe.inspect_codex_home(runtime_home)["ok"] is True

    residue = runtime_home / "analyze" / "history.jsonl"
    residue.write_text("safe synthetic prompt\n", encoding="utf-8")
    residue.chmod(0o600)
    rejected = probe.inspect_codex_home(runtime_home)
    assert rejected["ok"] is False
    assert rejected["persistent_session_or_history"] is True
    assert rejected["unknown_files"] == ["analyze/history.jsonl"]


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin extended ACL control")
def test_codex_home_probe_rejects_extended_acl_on_auth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import probe_m1_calls_access as probe

    home = tmp_path / "home"
    runtime_home = home / ".mango_local" / "codex-runtime"
    runtime_home.mkdir(parents=True, mode=0o700)
    runtime_home.chmod(0o700)
    auth = runtime_home / "auth.json"
    auth.write_text('{"synthetic":true}', encoding="utf-8")
    auth.chmod(0o600)
    subprocess.run(
        ["/bin/chmod", "+a", "everyone allow read", str(auth)],
        check=True,
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    rejected = probe.inspect_codex_home(runtime_home)

    assert rejected["ok"] is False
    assert rejected["owner_only_0700"] is True
    assert rejected["unsafe_files"] == ["auth.json"]


def test_duplicate_or_blank_env_values_are_not_accepted() -> None:
    with tempfile.TemporaryDirectory() as home:
        env_file = Path(home) / "worker.env"
        env_file.write_text(
            "MANGO_OFFICE_API_KEY=first\nMANGO_OFFICE_API_KEY=second\nMANGO_OFFICE_API_SALT=   \n",
            encoding="utf-8",
        )
        env_file.chmod(0o600)
        result = subprocess.run(
            [str(BOOTSTRAP), "check"], cwd=ROOT,
            env={**os.environ, "HOME": home, "MANGO_CALLS_ENV_FILE": str(env_file)},
            text=True, capture_output=True,
        )

    payload = json.loads(result.stdout)
    assert result.returncode != 0
    assert payload["mango_credentials_present"] is False


def test_env_parser_accepts_quoted_spaces_and_never_executes_shell(tmp_path: Path) -> None:
    parser = _load_env_parser()
    marker = tmp_path / "must-not-exist"
    valid = tmp_path / "valid.env"
    valid.write_text('PATH_VALUE="/Users/test/Mango Calls Resolve"\nEMPTY=\n', encoding="utf-8")
    assert parser.parse_env(valid) == {"PATH_VALUE": "/Users/test/Mango Calls Resolve", "EMPTY": ""}

    for index, value in enumerate((f'$(touch "{marker}")', "`touch marker`", "one two", "value # comment")):
        unsafe = tmp_path / f"unsafe-{index}.env"
        unsafe.write_text(f"VALUE={value}\n", encoding="utf-8")
        try:
            parser.parse_env(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe env accepted: {value}")
    assert not marker.exists()


def test_untrusted_symlink_config_cannot_execute_python(tmp_path: Path) -> None:
    home = tmp_path / "home"
    secrets = home / ".mango_secrets"
    secrets.mkdir(parents=True, mode=0o700)
    env_file = secrets / "mango_calls_m1_worker.env"
    env_file.write_text("MANGO_OFFICE_API_KEY=x\nMANGO_OFFICE_API_SALT=y\n", encoding="utf-8")
    env_file.chmod(0o600)
    marker = tmp_path / "executed"
    malicious = tmp_path / "malicious-python"
    malicious.write_text(f"#!/bin/zsh\ntouch {marker}\n", encoding="utf-8")
    malicious.chmod(0o700)
    real_config = tmp_path / "real-config.json"
    real_config.write_text(json.dumps({"python_executable": str(malicious)}), encoding="utf-8")
    real_config.chmod(0o600)
    config_link = tmp_path / "config.json"
    config_link.symlink_to(real_config)

    result = subprocess.run(
        [str(BOOTSTRAP), "check"], cwd=ROOT,
        env={**os.environ, "HOME": str(home), "MANGO_CALLS_CONFIG": str(config_link),
             "MANGO_CALLS_ENV_FILE": str(env_file)},
        text=True, capture_output=True,
    )

    assert result.returncode != 0
    assert not marker.exists()
    assert json.loads(result.stdout)["config_owner_only_and_valid"] is False


def test_operator_bash_blocks_are_fail_fast() -> None:
    def current_task_path(name: str) -> Path:
        matches = sorted((ROOT / "tasks").glob(f"_*/{name}"))
        assert len(matches) == 1, f"expected exactly one current task document for {name}: {matches}"
        return matches[0]

    documents = (
        (ROOT / "docs/M1_MANGO_CALLS_SPLIT_CUTOVER_RUNBOOK.md", 24),
        (HANDOFF / "README.md", 5),
        (HANDOFF / "M1_CODEX_PROMPT.md", 0),
        (current_task_path("2026-08-07_TZ_m1_calls_final_handoff.md"), 0),
        (current_task_path("2026-08-07_TZ_m1_calls_runtime_readiness.md"), 0),
        (current_task_path("2026-07-31_TZ_m1_calls_stage10_pilot.md"), 0),
    )
    for document, minimum_blocks in documents:
        lines = document.read_text(encoding="utf-8").splitlines()
        blocks = 0
        for index, line in enumerate(lines):
            if line.strip() in {"```bash", "```sh", "```zsh", "```shell"}:
                blocks += 1
                assert lines[index + 1].strip() == "set -euo pipefail", (
                    f"{document}:{index + 1} must start with fail-fast shell options"
                )
            if '"$(cat ' in line:
                assert lines[index + 1].lstrip().startswith("test "), (
                    f"{document}:{index + 1} must validate a path read from a file immediately"
                )
        assert blocks >= minimum_blocks, f"{document}: expected at least {minimum_blocks} shell blocks"


def test_runbook_sqlite_checks_assert_results() -> None:
    runbook = (ROOT / "docs/M1_MANGO_CALLS_SPLIT_CUTOVER_RUNBOOK.md").read_text(
        encoding="utf-8"
    )

    assert "sqlite3 " not in runbook.casefold()
    assert "wal_checkpoint" not in runbook
    assert runbook.count("assert_existing_single_link_database() {") == 1
    assert runbook.count("--check-sqlite") == 2
    assert 'assert_existing_single_link_database "$SOURCE_WORKING_DB"' in runbook
    inventory = '--inventory-root "$SOURCE_PIPELINE" --inventory-out "$SOURCE_INVENTORY"'
    verify = '--verify-inventory "$SOURCE_INVENTORY" --inventory-root "$SOURCE_PIPELINE"'
    selective_inventory = (
        '--selective-inventory-root "$SOURCE_PIPELINE" \\\n'
        '  --inventory-out "$SOURCE_INVENTORY"'
    )
    selective_verify = '--verify-selective-source "$SOURCE_INVENTORY"'
    wal_gate = 'test ! -s "$SOURCE_WORKING_WAL"'
    dry_run = '--source-inventory "$SOURCE_INVENTORY" --dry-run'
    transfer = '/usr/bin/rsync -aH --relative --from0'
    first_selective_verify = runbook.index(selective_verify)
    second_selective_verify = runbook.index(
        selective_verify, first_selective_verify + 1
    )
    assert runbook.index(inventory) < runbook.index(verify) < runbook.index(wal_gate)
    assert (
        runbook.index(wal_gate)
        < runbook.index(selective_inventory)
        < first_selective_verify
        < runbook.index(dry_run)
        < runbook.index(transfer)
        < second_selective_verify
    )
    assert "--from0" in runbook
    selective_transfer = runbook[
        runbook.index(selective_inventory):runbook.index("Последняя команда обязана")
    ]
    assert "--delete" not in selective_transfer
    assert 'assert_existing_single_link_file "$SOURCE_WORKING_WAL"' in runbook
    assert 'assert_existing_single_link_file "$SOURCE_WORKING_SHM"' in runbook
    source_stop = runbook[runbook.index("После снимка"):runbook.index("Если lock занят")]
    for label in (
        "com.mango.calls-two-processes",
        "com.mango.calls-process-a",
        "com.mango.calls-process-b",
        "com.mango.calls-capture",
        "com.mango.calls-pipeline",
        "com.mango.calls-watchdog",
        "com.mango.calls-publication-close-0600",
        "com.mango.calls-publication-close-0700",
        "com.mango.calls-publication-close-0800",
        "com.mango.calls-publication-alert-0830",
        "com.mango.calls-publication-status-0850",
    ):
        assert label in source_stop
    assert 'launchctl bootout "gui/$(id -u)/$label"' in source_stop
    assert '! launchctl print "gui/$(id -u)/$label"' in source_stop
    assert "mv \"$plist\" \"$SNAP/\"" in source_stop
    assert "active_calls_cron.txt" in source_stop
    assert "('process_a.lock', 'capture.lock', 'pipeline.lock', 'process_b.lock')" in source_stop


def test_remote_owner_local_repair_keeps_expansion_on_m1() -> None:
    runbook = (ROOT / "docs/M1_MANGO_CALLS_SPLIT_CUTOVER_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    matching = [
        line
        for line in runbook.splitlines()
        if line.startswith('ssh "$M1_HOST" ') and "OWNER_LOCAL=" in line
    ]
    assert len(matching) == 1

    command = shlex.split(matching[0])
    assert command[:2] == ["ssh", "$M1_HOST"]
    assert len(command) == 3
    remote = command[2]
    assert 'OWNER_LOCAL="$HOME/.mango_local"' in remote
    assert "\\\"" not in remote
    assert "/Users/" not in remote
    assert remote.index("test ! -L") < remote.index("stat -f %u ")
    assert remote.index("stat -f %u ") < remote.index("chmod 700")
    assert remote.index("chmod 700") < remote.index("stat -f %u:%Lp ")

    syntax = subprocess.run(
        ["zsh", "-n", "-c", remote],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    target_lines = [
        line
        for line in runbook.splitlines()
        if line.startswith('ssh "$M1_HOST" ') and "TARGET=~/.mango_local" in line
    ]
    assert len(target_lines) == 1
    capture = subprocess.run(
        [
            "zsh",
            "-c",
            "ssh() { test \"$#\" -eq 2; print -r -- \"$2\"; }\n"
            "M1_HOST=synthetic-host\n"
            "GENERATION=synthetic-generation\n"
            + target_lines[0],
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert capture.returncode == 0, capture.stderr
    target_remote = capture.stdout.strip()
    assert "mango_calls_transfers/synthetic-generation" in target_remote
    assert "&&" not in target_remote
    assert target_remote.index('test ! -e "$TARGET"') < target_remote.index(
        'mkdir -p "$TARGET"'
    )
    assert target_remote.index('mkdir -p "$TARGET"') < target_remote.index(
        'chmod 700 "$TARGET"'
    )
    target_syntax = subprocess.run(
        ["zsh", "-n", "-c", target_remote],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert target_syntax.returncode == 0, target_syntax.stderr
