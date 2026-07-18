from __future__ import annotations

import json
import plistlib
import shlex
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


def test_launchd_installer_renders_scheduled_a_and_demand_only_b(tmp_path: Path) -> None:
    plists = _render(
        tmp_path,
        "--process-a-interval-seconds",
        "600",
    )

    assert set(plists) == {"com.mango.calls-process-a", "com.mango.calls-process-b"}
    process_a = plists["com.mango.calls-process-a"]
    process_b = plists["com.mango.calls-process-b"]
    assert process_a["Label"] == "com.mango.calls-process-a"
    assert process_b["Label"] == "com.mango.calls-process-b"
    assert process_a["StartInterval"] == 600
    assert "StartInterval" not in process_b
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
            "--process-a-interval-seconds",
            "299",
        ],
        cwd=ROOT,
        text=True,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "at least 300 seconds" in result.stderr


def test_launchd_installer_rejects_process_b_interval(tmp_path: Path) -> None:
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
            "900",
        ],
        cwd=ROOT,
        text=True,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "demand-only" in result.stderr


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
    assert "/usr/bin/plutil -extract python_executable" in runner_text
    assert '"${python_executable}" "${root}/scripts/run_mango_calls_pipeline.py"' in runner_text
    assert "launchctl kickstart" in runner_text


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


@pytest.mark.parametrize("status", ["failed", "deferred", "locked"])
def test_process_a_does_not_start_b_without_explicit_success(
    tmp_path: Path, status: str
) -> None:
    config_path, env_path = _write_config(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "launchctl_args.txt"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        '#!/bin/zsh\nprintf "%s\\n" "$@" > "$CAPTURED_LAUNCHCTL"\n',
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o700)
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
else
  print -r -- '{{"status":"{status}"}}'
fi
''',
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_path), "process-a"],
        cwd=ROOT,
        env={
            **__import__("os").environ,
            "PATH": f"{fake_bin}:{__import__('os').environ.get('PATH', '')}",
            "CAPTURED_LAUNCHCTL": str(capture),
        },
        check=False,
    )

    assert result.returncode == 0
    assert not capture.exists()


def test_process_a_success_starts_demand_only_b(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    capture = tmp_path / "launchctl_args.txt"
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
else
  print -r -- 'diagnostic before result'
  print -r -- '{{'
  print -r -- '  "schema_version": "test_v1",'
  print -r -- '  "process": "process_a",'
  print -r -- '  "status": "ok",'
  print -r -- '  "counters": {{"nested": {{"status": "failed"}}}}'
  print -r -- '}}'
fi
''',
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    fake_launchctl = tmp_path / "launchctl"
    fake_launchctl.write_text(
        '#!/bin/zsh\nprintf "%s\\n" "$@" > "$CAPTURED_LAUNCHCTL"\n',
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o700)

    # The wrapper uses the absolute platform path. Keep the test hermetic by
    # copying it and substituting only the command path in the copy.
    runner = tmp_path / "runner.sh"
    runner.write_text(
        RUNNER.read_text(encoding="utf-8").replace("/bin/launchctl", str(fake_launchctl)),
        encoding="utf-8",
    )
    runner.chmod(0o700)
    result = subprocess.run(
        [str(runner), str(config_path), str(env_path), "process-a"],
        cwd=ROOT,
        env={**__import__("os").environ, "CAPTURED_LAUNCHCTL": str(capture)},
        check=False,
    )

    assert result.returncode == 0
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "kickstart",
        f"gui/{__import__('os').getuid()}/com.mango.calls-process-b",
    ]


def test_process_a_reports_kickstart_failure(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
else
  print -r -- '{{"status":"ok"}}'
fi
''',
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    fake_launchctl = tmp_path / "launchctl"
    fake_launchctl.write_text("#!/bin/zsh\nexit 9\n", encoding="utf-8")
    fake_launchctl.chmod(0o700)
    runner = tmp_path / "runner.sh"
    runner.write_text(
        RUNNER.read_text(encoding="utf-8").replace("/bin/launchctl", str(fake_launchctl)),
        encoding="utf-8",
    )
    runner.chmod(0o700)

    result = subprocess.run(
        [str(runner), str(config_path), str(env_path), "process-a"],
        cwd=ROOT,
        check=False,
    )

    assert result.returncode == 9


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


def test_failed_upgrade_restores_loaded_incumbent_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, env_path = _write_config(tmp_path)
    out_dir = tmp_path / "launchd"
    out_dir.mkdir()
    installer = _load_installer()
    domain = f"gui/{installer.os.getuid()}"
    paths = {
        label: out_dir / f"{label}.plist"
        for label in (installer.LABEL_A, installer.LABEL_B)
    }
    old_bytes = {
        installer.LABEL_A: b"old-a",
        installer.LABEL_B: b"old-b",
    }
    for label, path in paths.items():
        path.write_bytes(old_bytes[label])
    loaded = {f"{domain}/{installer.LABEL_A}", f"{domain}/{installer.LABEL_B}"}
    failed_a_once = False

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal failed_a_once
        del kwargs
        if command[:2] == ["launchctl", "print"]:
            return subprocess.CompletedProcess(command, 0 if command[2] in loaded else 1)
        if command[:2] == ["launchctl", "bootout"]:
            loaded.discard(command[2])
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["launchctl", "bootstrap"]:
            label = Path(command[3]).stem
            if label == installer.LABEL_A and not failed_a_once:
                failed_a_once = True
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="failed")
            loaded.add(f"{domain}/{label}")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
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
    ) == 1
    assert loaded == {f"{domain}/{installer.LABEL_A}", f"{domain}/{installer.LABEL_B}"}
    assert {label: path.read_bytes() for label, path in paths.items()} == old_bytes


def test_failed_bootout_restores_loaded_incumbent_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, env_path = _write_config(tmp_path)
    out_dir = tmp_path / "launchd"
    out_dir.mkdir()
    installer = _load_installer()
    domain = f"gui/{installer.os.getuid()}"
    paths = {
        label: out_dir / f"{label}.plist"
        for label in (installer.LABEL_A, installer.LABEL_B)
    }
    old_bytes = {installer.LABEL_A: b"old-a", installer.LABEL_B: b"old-b"}
    for label, path in paths.items():
        path.write_bytes(old_bytes[label])
    loaded = {f"{domain}/{installer.LABEL_A}", f"{domain}/{installer.LABEL_B}"}
    failed_b_bootout_once = False

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal failed_b_bootout_once
        del kwargs
        if command[:2] == ["launchctl", "print"]:
            return subprocess.CompletedProcess(command, 0 if command[2] in loaded else 1)
        if command[:2] == ["launchctl", "bootout"]:
            if command[2].endswith(installer.LABEL_B) and not failed_b_bootout_once:
                failed_b_bootout_once = True
                return subprocess.CompletedProcess(command, 1)
            loaded.discard(command[2])
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["launchctl", "bootstrap"]:
            label = Path(command[3]).stem
            loaded.add(f"{domain}/{label}")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
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
    ) == 1
    assert loaded == {f"{domain}/{installer.LABEL_A}", f"{domain}/{installer.LABEL_B}"}
    assert {label: path.read_bytes() for label, path in paths.items()} == old_bytes
