from __future__ import annotations

import json
import plistlib
import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_mango_calls_two_processes_service.py"
RUNNER = ROOT / "scripts" / "run_mango_calls_process.sh"


def _load_installer():
    spec = importlib.util.spec_from_file_location("install_mango_calls_two_processes_service", INSTALLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "config.json"
    env_path = tmp_path / "mango.env"
    pipeline_root = tmp_path / "pipeline"
    staging = tmp_path / "staging"
    config_path.write_text(
        json.dumps(
            {
                "pipeline_root": str(pipeline_root),
                "timeline_db": str(staging / "customer_timeline_staging.sqlite"),
                "timeline_allowed_root": str(staging),
                "python_executable": sys.executable,
                "codex_binary": sys.executable,
                "codex_home_root": str(tmp_path / "codex_home"),
            }
        ),
        encoding="utf-8",
    )
    env_path.write_text("MANGO_OFFICE_API_KEY=x\nMANGO_OFFICE_API_SALT=y\n", encoding="utf-8")
    return config_path, env_path


def _render(tmp_path: Path, *extra: str) -> dict[str, dict[str, object]]:
    config_path, env_path = _write_config(tmp_path)
    out_dir = tmp_path / "launchd"
    subprocess.run(
        [
            sys.executable,
            str(INSTALLER),
            "--config",
            str(config_path),
            "--env-file",
            str(env_path),
            "--out-dir",
            str(out_dir),
            *extra,
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return {
        path.stem: plistlib.loads(path.read_bytes())
        for path in sorted(out_dir.glob("com.mango.calls-process-*.plist"))
    }


def test_launchd_installer_renders_two_independent_process_plists(tmp_path: Path) -> None:
    plists = _render(
        tmp_path,
        "--process-a-interval-seconds",
        "600",
        "--process-b-interval-seconds",
        "900",
    )

    assert set(plists) == {"com.mango.calls-process-a", "com.mango.calls-process-b"}
    process_a = plists["com.mango.calls-process-a"]
    process_b = plists["com.mango.calls-process-b"]
    assert process_a["Label"] == "com.mango.calls-process-a"
    assert process_b["Label"] == "com.mango.calls-process-b"
    assert process_a["StartInterval"] == 600
    assert process_b["StartInterval"] == 900
    assert process_a["ProgramArguments"][-1] == "process-a"
    assert process_b["ProgramArguments"][-1] == "process-b"
    assert "cycle" not in process_a["ProgramArguments"]
    assert "cycle" not in process_b["ProgramArguments"]
    assert process_a["StandardOutPath"] != process_b["StandardOutPath"]
    assert process_a["StandardErrorPath"] != process_b["StandardErrorPath"]


def test_launchd_installer_rejects_sub_300_second_intervals(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(INSTALLER),
            "--config",
            str(config_path),
            "--env-file",
            str(env_path),
            "--out-dir",
            str(tmp_path / "launchd"),
            "--process-b-interval-seconds",
            "299",
        ],
        cwd=ROOT,
        text=True,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "at least 300 seconds" in result.stderr


def test_process_b_launchd_path_does_not_use_cycle_or_asr_runner(tmp_path: Path) -> None:
    plists = _render(tmp_path)
    process_b = plists["com.mango.calls-process-b"]
    args = process_b["ProgramArguments"]

    assert args[1] == str(RUNNER)
    assert args[-1] == "process-b"
    assert "cycle" not in args
    runner_text = RUNNER.read_text(encoding="utf-8").lower()
    assert "asr" not in runner_text
    assert "resolve" not in runner_text
    assert "analyze" not in runner_text
    assert "json.load" in runner_text
    assert 'exec "${python_executable}"' in runner_text


def test_runner_executes_configured_python(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    captured = tmp_path / "python_args.txt"
    fake_python = tmp_path / "configured-python"
    fake_python.write_text('#!/bin/zsh\nprintf "%s\\n" "$@" > "$CAPTURED"\n', encoding="utf-8")
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    subprocess.run(
        [str(RUNNER), str(config_path), str(env_path), "process-b"],
        cwd=ROOT,
        env={**__import__("os").environ, "CAPTURED": str(captured)},
        check=True,
    )

    args = captured.read_text(encoding="utf-8").splitlines()
    assert args[-1] == "process-b"
    assert "run_mango_calls_pipeline.py" in args[0]


def test_install_boots_out_old_loaded_label_without_deleting_plist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, env_path = _write_config(tmp_path)
    out_dir = tmp_path / "launchd"
    installer = _load_installer()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(command)
        if command[:2] == ["launchctl", "print"]:
            return subprocess.CompletedProcess(command, 0 if command[2].endswith("/com.mango.calls-two-processes") else 1)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(installer.subprocess, "run", fake_run)

    assert installer.main(
        [
            "--config",
            str(config_path),
            "--env-file",
            str(env_path),
            "--out-dir",
            str(out_dir),
            "--install",
        ]
    ) == 0

    domain = f"gui/{installer.os.getuid()}"
    assert ["launchctl", "bootout", f"{domain}/com.mango.calls-two-processes"] in calls
    assert ["launchctl", "bootstrap", domain, str(out_dir / "com.mango.calls-process-a.plist")] in calls
    assert ["launchctl", "bootstrap", domain, str(out_dir / "com.mango.calls-process-b.plist")] in calls


def test_partial_install_rolls_back_process_a(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    out_dir = tmp_path / "launchd"
    installer = _load_installer()
    domain = f"gui/{installer.os.getuid()}"
    old_target = f"{domain}/com.mango.calls-two-processes"
    loaded: set[str] = {old_target}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        if command[:2] == ["launchctl", "print"]:
            return subprocess.CompletedProcess(command, 0 if command[2] in loaded else 1)
        if command[:2] == ["launchctl", "bootstrap"]:
            label = Path(command[3]).stem
            if label.endswith("process-b"):
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="failed")
            loaded.add(f"gui/{installer.os.getuid()}/{label}")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["launchctl", "bootout"]:
            loaded.discard(command[2])
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    result = installer.main(
        [
            "--config",
            str(config_path),
            "--env-file",
            str(env_path),
            "--out-dir",
            str(out_dir),
            "--install",
        ]
    )
    assert result == 1
    assert loaded == {old_target}
