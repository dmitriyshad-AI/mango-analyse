from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts/bootstrap_m1_mango_calls.sh"
HANDOFF = ROOT / "docs/m1_calls_handoff_20260801"
ENV_PARSER = ROOT / "scripts/mango_calls_env.py"


def _load_env_parser():
    spec = importlib.util.spec_from_file_location("mango_calls_env", ENV_PARSER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_plan_is_safe_and_complete() -> None:
    result = subprocess.run(
        [str(BOOTSTRAP), "plan"], cwd=ROOT, text=True, capture_output=True, check=True
    )
    payload = json.loads(result.stdout)

    assert payload["starts_services"] is False
    assert payload["runs_asr"] is False
    assert payload["runs_resolve_analyze"] is False
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
    assert '&& "$tallanto_export" == true' in source
    assert '&& "$google_config_valid" == true' in source
    assert '&& "$yandex" == true' in source
    assert '"developer_profile_ready":%s' in source
    assert '&& "$skills" == true' not in source
    assert '&& "$pipeline_root_owner_only" == true' in source
    assert '&& "$pipeline_root_under_owner_local" == true' in source
    assert '&& "$pipeline_root_matches_env" == true' in source
    assert '"pipeline_root_under_owner_local":%s' in source
    assert '"pipeline_root_matches_env":%s' in source
    assert '"host_preflight_passed":%s' in source
    assert '"runtime_ready":false' in source
    assert '&& "$disk_space_ok" == true' in source
    assert '&& "$conflicting_services_loaded" == false' in source
    assert '&& "$pipeline_lock_held" == false' in source
    assert '${GOOGLE_APPLICATION_CREDENTIALS:-' not in source
    assert 'manifest["local_skills"]' in source
    assert '== "codex-cli 0.142.3"' in source

    config = json.loads((HANDOFF / "config.m1.example.json").read_text(encoding="utf-8"))
    assert "Projects/Mango analyse" not in config["pipeline_root"]
    assert config["pipeline_root"] == "<HOME>/.mango_local/mango_calls_two_processes"


def test_google_is_optional_but_partial_google_config_is_blocked() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")

    assert '[[ -z "$google_path" && -z "$google_folder_id" ]]' in source
    assert '[[ "$google" == true ]] && google_config_valid=true' in source
    env_example = (HANDOFF / "mango_calls_m1_worker.env.example").read_text(encoding="utf-8")
    assert "MANGO_CALLS_GOOGLE_DRIVE_FOLDER_ID=\n" in env_example
    assert "GOOGLE_APPLICATION_CREDENTIALS=\n" in env_example


def test_handoff_examples_contain_no_secret_values() -> None:
    env_lines = (HANDOFF / "mango_calls_m1_worker.env.example").read_text(encoding="utf-8").splitlines()
    secret_keys = {
        "MANGO_OFFICE_API_KEY",
        "MANGO_OFFICE_API_SALT",
        "MANGO_CALLS_GOOGLE_DRIVE_FOLDER_ID",
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
    documents = (
        (ROOT / "docs/M1_MANGO_CALLS_SPLIT_CUTOVER_RUNBOOK.md", 24),
        (HANDOFF / "README.md", 5),
        (HANDOFF / "M1_CODEX_PROMPT.md", 0),
        (ROOT / "tasks/_running/2026-08-07_TZ_m1_calls_final_handoff.md", 0),
        (ROOT / "tasks/_inbox_codex/2026-08-07_TZ_m1_calls_runtime_readiness.md", 0),
        (ROOT / "tasks/_inbox_codex/2026-07-31_TZ_m1_calls_stage10_pilot.md", 0),
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
