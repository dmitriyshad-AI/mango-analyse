from __future__ import annotations

import json
import fcntl
import os
import plistlib
import shutil
import shlex
import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_mango_calls_two_processes_service.py"
RUNNER = ROOT / "scripts" / "run_mango_calls_process.sh"
LEGACY_RUNNER = ROOT / "scripts" / "run_mango_calls_cycle.sh"


def _verified_yandex_target(tmp_path: Path, name: str = "out") -> Path:
    target = tmp_path / name
    target.mkdir()
    (target / ".mango_calls_yandex_target").write_text("mango-calls-yandex-v1\n", encoding="utf-8")
    return target


def _load_installer():
    spec = importlib.util.spec_from_file_location("install_mango_calls_two_processes_service", INSTALLER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plist_writer_rejects_fixed_temp_symlink_attack(tmp_path: Path) -> None:
    installer = _load_installer()
    target = tmp_path / "service.plist"
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"must remain unchanged")
    legacy_temp = target.with_suffix(".plist.tmp")
    legacy_temp.symlink_to(victim)

    installer._write_plist(target, {"Label": "synthetic.safe"})

    assert victim.read_bytes() == b"must remain unchanged"
    assert legacy_temp.is_symlink()
    assert target.is_file() and not target.is_symlink()
    assert (target.stat().st_mode & 0o777) == 0o600


def _allow_test_install_path(
    monkeypatch: pytest.MonkeyPatch, installer: object, out_dir: Path
) -> None:
    monkeypatch.setattr(
        installer,
        "_standard_plist",
        lambda label: out_dir.resolve() / f"{label}.plist",
    )
    monkeypatch.setattr(
        installer,
        "_validate_process_a_install_authority",
        lambda *_args, **_kwargs: None,
    )


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "config.json"
    env_path = tmp_path / "mango.env"
    pipeline_root = tmp_path / ".mango_local" / "pipeline"
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
                "require_cutover_authority": False,
                "strict_ready_provenance": False,
            }
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o600)
    env_path.write_text(
        f"MANGO_OFFICE_API_KEY=x\nMANGO_OFFICE_API_SALT=y\nMANGO_CALLS_PIPELINE_ROOT={pipeline_root}\n",
        encoding="utf-8",
    )
    env_path.chmod(0o600)
    return config_path, env_path


def _copy_runtime_reader_modules(target_root: Path) -> None:
    for relative in (
        "src/mango_mvp/__init__.py",
        "src/mango_mvp/productization/__init__.py",
        "src/mango_mvp/productization/owner_only_io.py",
        "src/mango_mvp/productization/mango_calls_config.py",
    ):
        source = ROOT / relative
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _clean_git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "clean-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    scripts = repo / "scripts"
    scripts.mkdir()
    runner = scripts / RUNNER.name
    runner.write_text(RUNNER.read_text(encoding="utf-8"), encoding="utf-8")
    runner.chmod(0o700)
    (scripts / "mango_calls_env.py").write_text(
        (ROOT / "scripts" / "mango_calls_env.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    _copy_runtime_reader_modules(repo)
    subprocess.run(["git", "add", "scripts", "src"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "test"], cwd=repo, check=True)
    return repo, subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _worker_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {**os.environ, "HOME": str(tmp_path), **extra}


def _copy_runner(tmp_path: Path, launchctl: Path) -> Path:
    scripts = tmp_path / "runner-repo" / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / RUNNER.name
    runner.write_text(
        RUNNER.read_text(encoding="utf-8").replace("/bin/launchctl", str(launchctl)),
        encoding="utf-8",
    )
    runner.chmod(0o700)
    (scripts / "mango_calls_env.py").write_text(
        (ROOT / "scripts" / "mango_calls_env.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    _copy_runtime_reader_modules(scripts.parent)
    return runner


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
    assert process_a["Umask"] == process_b["Umask"] == 63


def test_strict_config_requires_explicit_installer_topology(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(require_cutover_authority=True, strict_ready_provenance=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    installer = _load_installer()

    with pytest.raises(RuntimeError, match="requires --fast-service"):
        installer.main(
            [
                "--config",
                str(config_path),
                "--env-file",
                str(env_path),
                "--out-dir",
                str(tmp_path / "launchd"),
            ]
        )


def test_missing_strict_flags_require_explicit_installer_topology(
    tmp_path: Path,
) -> None:
    config_path, env_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("require_cutover_authority")
    config.pop("strict_ready_provenance")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    installer = _load_installer()

    with pytest.raises(RuntimeError, match="requires --fast-service"):
        installer.main(
            [
                "--config",
                str(config_path),
                "--env-file",
                str(env_path),
                "--out-dir",
                str(tmp_path / "launchd"),
            ]
        )


def test_fast_service_renders_exact_publication_schedule_without_execute(
    tmp_path: Path,
) -> None:
    config_path, env_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(require_cutover_authority=True, strict_ready_provenance=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
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
            "--fast-service",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    plists = {
        path.stem: plistlib.loads(path.read_bytes())
        for path in out_dir.glob("*.plist")
    }
    expected = {
        "com.mango.calls-publication-close-0600": (6, 0, "publication-close"),
        "com.mango.calls-publication-close-0700": (7, 0, "publication-close"),
        "com.mango.calls-publication-close-0800": (8, 0, "publication-close"),
        "com.mango.calls-publication-alert-0830": (8, 30, "publication-alert"),
        "com.mango.calls-publication-status-0850": (8, 50, "publication-status"),
    }
    for label, (hour, minute, command) in expected.items():
        assert plists[label]["StartCalendarInterval"] == [
            {"Hour": hour, "Minute": minute}
        ]
        assert plists[label]["ProgramArguments"][-1] == command
        assert "--execute" not in json.dumps(plists[label])
    assert all(payload["Umask"] == 63 for payload in plists.values())
    assert plists["com.mango.calls-pipeline"]["StartCalendarInterval"] == [
        {"Minute": 7},
        {"Minute": 37},
    ]
    assert {
        plists[label]["ProgramArguments"][-1]
        for label in (
            "com.mango.calls-capture",
            "com.mango.calls-pipeline",
            "com.mango.calls-process-b",
            "com.mango.calls-watchdog",
        )
    } == {
        "capture-worker",
        "pipeline-worker",
        "process-b-worker",
        "watchdog-worker",
    }
    runner = RUNNER.read_text(encoding="utf-8")
    assert "current-plan" in runner
    assert "publish_daily_mango_calls_google.py" not in runner
    assert "UPLOAD_MANGO_DAILY_REPORT" not in runner


def test_render_only_without_output_refuses_before_runtime_mutation(
    tmp_path: Path,
) -> None:
    config_path, env_path = _write_config(tmp_path)
    pipeline = Path(json.loads(config_path.read_text(encoding="utf-8"))["pipeline_root"])

    result = subprocess.run(
        [
            sys.executable,
            str(INSTALLER),
            "--config",
            str(config_path),
            "--env-file",
            str(env_path),
            "--fast-service",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "explicit --out" in result.stderr
    assert not pipeline.exists()


def test_launchd_installer_process_a_only_never_renders_b(tmp_path: Path) -> None:
    plists = _render(tmp_path, "--process-a-only", "--process-a-interval-seconds", "600")
    assert set(plists) == {"com.mango.calls-process-a"}
    assert plists["com.mango.calls-process-a"]["ProgramArguments"][-1] == "process-a-worker"


def test_launchd_installer_process_b_only_never_renders_a(tmp_path: Path) -> None:
    plists = _render(tmp_path, "--process-b-only", "--process-b-interval-seconds", "600")
    assert set(plists) == {"com.mango.calls-process-b"}
    assert plists["com.mango.calls-process-b"]["StartInterval"] == 600
    assert plists["com.mango.calls-process-b"]["ProgramArguments"][-1] == "process-b-pull"


def test_launchd_installer_rejects_both_single_process_modes(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    result = subprocess.run([
        sys.executable, str(INSTALLER), "--config", str(config_path), "--env-file", str(env_path),
        "--process-a-only", "--process-b-only", "--out-dir", str(tmp_path / "launchd"),
    ], cwd=ROOT, check=False)
    assert result.returncode != 0


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
    assert "--stages" not in runner_text
    assert "mango_mvp.cli" not in runner_text
    assert "mango_mvp.productization.mango_calls_config" in runner_text
    assert "/usr/bin/plutil" not in runner_text
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


def test_runner_rejects_readable_secret_env_before_source(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    env_path.chmod(0o644)
    result = subprocess.run([str(RUNNER), str(config_path), str(env_path), "process-b"], cwd=ROOT, check=False)
    assert result.returncode == 2


def test_successful_process_b_never_runs_daily_export_inline(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    captured = tmp_path / "python_args.txt"
    out = _verified_yandex_target(tmp_path, "yandex")
    env_path.write_text(
        env_path.read_text(encoding="utf-8") + f"MANGO_CALLS_DAILY_EXPORT_OUT={out}\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
printf -- "--call--\\n" >> "$CAPTURED"
printf "%s\\n" "$@" >> "$CAPTURED"
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
elif [[ "$1" == *run_mango_calls_pipeline.py ]]; then
  print -r -- '{{"status":"ok","stop_reason":""}}'
fi
''',
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    subprocess.run([str(RUNNER), str(config_path), str(env_path), "process-b"], cwd=ROOT,
                   env={**__import__("os").environ, "CAPTURED": str(captured)}, check=True)

    calls = captured.read_text(encoding="utf-8").split("--call--\n")
    assert sum("run_mango_calls_pipeline.py" in call for call in calls) == 1
    assert all("export_daily_mango_calls_resolve.py" not in call for call in calls)


def test_runner_does_not_inherit_missing_worker_env_values(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    captured = tmp_path / "python_args.txt"
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        '#!/bin/zsh\nprintf "%s\\n" "$@" >> "$CAPTURED"\n'
        '[[ "$1" == *run_mango_calls_pipeline.py ]] && print -r -- \'{"status":"idle"}\'\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_path), "process-b"], cwd=ROOT,
        env={**__import__("os").environ, "CAPTURED": str(captured),
             "MANGO_CALLS_DAILY_EXPORT_OUT": str(tmp_path / "must-not-be-used")},
        check=False,
    )

    assert result.returncode == 0
    assert "export_daily_mango_calls_resolve.py" not in captured.read_text(encoding="utf-8")


def test_process_b_never_runs_google_publisher_even_with_legacy_config(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    captured = tmp_path / "python_args.txt"
    out = _verified_yandex_target(tmp_path)
    env_path.write_text(
        env_path.read_text(encoding="utf-8")
        + f"MANGO_CALLS_DAILY_EXPORT_OUT={out}\n"
        + "MANGO_CALLS_GOOGLE_DRIVE_FOLDER_ID=folder_123456789\n"
        + f"GOOGLE_APPLICATION_CREDENTIALS={tmp_path / 'credentials.json'}\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
printf -- "--call--\\n" >> "$CAPTURED"
printf "%s\\n" "$@" >> "$CAPTURED"
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
elif [[ "$1" == *run_mango_calls_pipeline.py ]]; then
  print -r -- '{{"status":"ok","stop_reason":""}}'
fi
''', encoding="utf-8")
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    subprocess.run([str(RUNNER), str(config_path), str(env_path), "process-b"], cwd=ROOT,
                   env={**__import__("os").environ, "CAPTURED": str(captured)}, check=True)

    calls = captured.read_text(encoding="utf-8").split("--call--\n")
    assert all("export_daily_mango_calls_resolve.py" not in call for call in calls)
    assert all("publish_daily_mango_calls_google.py" not in call for call in calls)


def test_process_b_ignores_legacy_partial_google_config(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    env_path.write_text(
        env_path.read_text(encoding="utf-8")
        + f"MANGO_CALLS_DAILY_EXPORT_OUT={tmp_path / 'out'}\n"
        + "MANGO_CALLS_GOOGLE_DRIVE_FOLDER_ID=folder_123456789\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
elif [[ "$1" == *run_mango_calls_pipeline.py ]]; then
  print -r -- '{{"status":"ok","stop_reason":""}}'
fi
''', encoding="utf-8")
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run([str(RUNNER), str(config_path), str(env_path), "process-b"], cwd=ROOT, check=False)

    assert result.returncode == 0


@pytest.mark.parametrize("status,reason", [("locked", "timeline_writer_locked"), ("deferred", "network"), ("idle", "drop_missing")])
def test_process_b_non_success_does_not_run_daily_export(tmp_path: Path, status: str, reason: str) -> None:
    config_path, env_path = _write_config(tmp_path)
    captured = tmp_path / "python_args.txt"
    env_path.write_text(env_path.read_text(encoding="utf-8") + f"MANGO_CALLS_DAILY_EXPORT_OUT={tmp_path / 'out'}\n", encoding="utf-8")
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
printf "%s\\n" "$@" >> "$CAPTURED"
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
elif [[ "$1" == *run_mango_calls_pipeline.py ]]; then
  print -r -- '{{"status":"{status}","stop_reason":"{reason}"}}'
fi
''', encoding="utf-8")
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run([str(RUNNER), str(config_path), str(env_path), "process-b"], cwd=ROOT,
                            env={**__import__("os").environ, "CAPTURED": str(captured)}, check=False)

    assert result.returncode == 0
    assert "export_daily_mango_calls_resolve.py" not in captured.read_text(encoding="utf-8")


def test_process_b_idle_unchanged_does_not_publish_inline(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    captured = tmp_path / "python_args.txt"
    out = _verified_yandex_target(tmp_path)
    env_path.write_text(env_path.read_text(encoding="utf-8") + f"MANGO_CALLS_DAILY_EXPORT_OUT={out}\n", encoding="utf-8")
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
printf "%s\\n" "$@" >> "$CAPTURED"
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
elif [[ "$1" == *run_mango_calls_pipeline.py ]]; then
  print -r -- '{{"status":"idle","stop_reason":"drop_unchanged"}}'
fi
''', encoding="utf-8")
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    subprocess.run([str(RUNNER), str(config_path), str(env_path), "process-b"], cwd=ROOT,
                   env={**__import__("os").environ, "CAPTURED": str(captured)}, check=True)

    assert "export_daily_mango_calls_resolve.py" not in captured.read_text(encoding="utf-8")


def test_process_b_wrapper_returns_pipeline_rc_without_parsing_publication(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    env_path.write_text(env_path.read_text(encoding="utf-8") + f"MANGO_CALLS_DAILY_EXPORT_OUT={tmp_path / 'out'}\n", encoding="utf-8")
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
else
  print -r -- 'not-json'
fi
''', encoding="utf-8")
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run([str(RUNNER), str(config_path), str(env_path), "process-b"], cwd=ROOT, check=False)

    assert result.returncode == 0


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
    runner = _copy_runner(tmp_path, fake_launchctl)

    result = subprocess.run(
        [str(runner), str(config_path), str(env_path), "process-a"],
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


def test_process_a_partial_without_ready_drop_does_not_start_b(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    capture = tmp_path / "launchctl_args.txt"
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
else
  print -r -- '{{"status":"partial","downstream_ready":false}}'
  exit 1
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
        '#!/bin/zsh\nprintf "%s\\n" "$@" > "$CAPTURED_LAUNCHCTL"\n', encoding="utf-8"
    )
    fake_launchctl.chmod(0o700)
    runner = _copy_runner(tmp_path, fake_launchctl)

    result = subprocess.run(
        [str(runner), str(config_path), str(env_path), "process-a"],
        cwd=ROOT,
        env={**__import__("os").environ, "CAPTURED_LAUNCHCTL": str(capture)},
        check=False,
    )

    assert result.returncode == 1
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
  print -r -- '  "downstream_ready": true,'
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
    runner = _copy_runner(tmp_path, fake_launchctl)
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


@pytest.mark.parametrize("process_status", ["ok", "partial"])
def test_process_a_worker_never_publishes_inline(
    tmp_path: Path, process_status: str
) -> None:
    config_path, env_path = _write_config(tmp_path)
    captured = tmp_path / "python_args.txt"
    launchctl_capture = tmp_path / "launchctl_args.txt"
    out = tmp_path / "out"
    out.mkdir()
    (out / ".mango_calls_yandex_target").write_text("mango-calls-yandex-v1\n", encoding="utf-8")
    env_path.write_text(
        env_path.read_text(encoding="utf-8")
        + f"MANGO_CALLS_DAILY_EXPORT_OUT={out}\n",
        encoding="utf-8",
    )
    clean_repo, head = _clean_git_repo(tmp_path)
    env_path.write_text(env_path.read_text(encoding="utf-8")
                        + f"MANGO_CALLS_EXPECTED_CODE_SHA={head}\n", encoding="utf-8")
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
printf -- "--call--\\n" >> "$CAPTURED"
printf "%s\\n" "$@" >> "$CAPTURED"
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
elif [[ "$1" == *run_mango_calls_pipeline.py ]]; then
  print -r -- '{{"status":"{process_status}","downstream_ready":true}}'
  [[ "{process_status}" == "ok" ]] || exit 1
fi
''', encoding="utf-8")
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    fake_launchctl = tmp_path / "launchctl"
    fake_launchctl.write_text('#!/bin/zsh\nprintf "%s\\n" "$@" > "$CAPTURED_LAUNCHCTL"\n', encoding="utf-8")
    fake_launchctl.chmod(0o700)
    runner = clean_repo / "scripts" / RUNNER.name

    result = subprocess.run(
        [str(runner), str(config_path), str(env_path), "process-a-worker"], cwd=ROOT,
        env=_worker_env(tmp_path, CAPTURED=str(captured), CAPTURED_LAUNCHCTL=str(launchctl_capture)),
        check=False,
    )

    calls = captured.read_text(encoding="utf-8").split("--call--\n")
    exports = [call for call in calls if "export_daily_mango_calls_resolve.py" in call]
    ready = str(tmp_path / ".mango_local" / "pipeline" / "drop" / "mango_calls_ready.sqlite")
    assert result.returncode == (0 if process_status == "ok" else 1)
    assert not exports
    assert not launchctl_capture.exists()


def test_process_a_worker_does_not_touch_unverified_yandex_target(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    out = tmp_path / "ordinary-local-folder"
    out.mkdir()
    clean_repo, head = _clean_git_repo(tmp_path)
    env_path.write_text(
        env_path.read_text(encoding="utf-8")
        + f"MANGO_CALLS_EXPECTED_CODE_SHA={head}\n"
        + f"MANGO_CALLS_DAILY_EXPORT_OUT={out}\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
elif [[ "$1" == *run_mango_calls_pipeline.py ]]; then
  print -r -- '{{"status":"ok","downstream_ready":true}}'
fi
''',
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [str(clean_repo / "scripts" / RUNNER.name), str(config_path), str(env_path), "process-a-worker"],
        cwd=ROOT,
        env=_worker_env(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "yandex_publish_target_not_verified" not in result.stderr


def test_process_a_worker_does_not_require_yandex_publish_target(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    clean_repo, head = _clean_git_repo(tmp_path)
    env_path.write_text(
        env_path.read_text(encoding="utf-8") + f"MANGO_CALLS_EXPECTED_CODE_SHA={head}\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
elif [[ "$1" == *run_mango_calls_pipeline.py ]]; then
  print -r -- '{{"status":"ok","downstream_ready":true}}'
fi
''',
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [str(clean_repo / "scripts" / RUNNER.name), str(config_path), str(env_path), "process-a-worker"],
        cwd=ROOT,
        env=_worker_env(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "m1_yandex_publish_target_missing" not in result.stderr


def test_legacy_process_b_keeps_existing_unmarked_export_path(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    ordinary_out = tmp_path / "legacy-export"
    ordinary_out.mkdir()
    env_path.write_text(
        env_path.read_text(encoding="utf-8") + f"MANGO_CALLS_DAILY_EXPORT_OUT={ordinary_out}\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
elif [[ "$1" == *run_mango_calls_pipeline.py ]]; then
  print -r -- '{{"status":"ok","stop_reason":""}}'
elif [[ "$1" == *export_daily_mango_calls_resolve.py ]]; then
  exit 0
fi
''',
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_path), "process-b"],
        cwd=ROOT, text=True, capture_output=True,
    )

    assert result.returncode == 0
    assert "yandex_publish_target_not_verified" not in result.stderr


def test_process_wrapper_rejects_pipeline_root_inside_repository(tmp_path: Path) -> None:
    config_path, env_file = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pipeline_root"] = str(ROOT / "product_data" / "forbidden-runtime")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    env_file.write_text(
        env_file.read_text(encoding="utf-8").replace(
            str(tmp_path / ".mango_local" / "pipeline"), config["pipeline_root"]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_file), "process-a-worker"],
        cwd=ROOT, env=_worker_env(tmp_path), text=True, capture_output=True,
    )

    assert result.returncode == 2
    assert "pipeline_root_outside_owner_local_root_or_symlink" in result.stderr


def test_process_wrapper_rejects_pipeline_root_symlink(tmp_path: Path) -> None:
    config_path, env_file = _write_config(tmp_path)
    real_root = tmp_path / ".mango_local" / "real-pipeline"
    real_root.mkdir(parents=True)
    link_root = tmp_path / ".mango_local" / "pipeline"
    link_root.symlink_to(real_root)

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_file), "process-a-worker"],
        cwd=ROOT, env=_worker_env(tmp_path), text=True, capture_output=True,
    )

    assert result.returncode == 2
    assert "pipeline_root_outside_owner_local_root_or_symlink" in result.stderr


def test_process_wrapper_rejects_symlinked_owner_local_root(tmp_path: Path) -> None:
    config_path, env_file = _write_config(tmp_path)
    cloud_root = tmp_path / "cloud-runtime"
    pipeline_root = cloud_root / "pipeline"
    pipeline_root.mkdir(parents=True)
    (tmp_path / ".mango_local").symlink_to(cloud_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pipeline_root"] = str(pipeline_root)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    env_file.write_text(
        f"MANGO_OFFICE_API_KEY=x\nMANGO_OFFICE_API_SALT=y\nMANGO_CALLS_PIPELINE_ROOT={pipeline_root}\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_file), "process-a-worker"],
        cwd=ROOT, env=_worker_env(tmp_path), text=True, capture_output=True,
    )

    assert result.returncode == 2
    assert "pipeline_root_outside_owner_local_root_or_symlink" in result.stderr


def test_process_wrapper_rejects_runtime_nested_in_git_repo(tmp_path: Path) -> None:
    config_path, env_file = _write_config(tmp_path)
    pipeline_root = tmp_path / ".mango_local" / "pipeline"
    (pipeline_root / ".git").mkdir(parents=True)

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_file), "process-a-worker"],
        cwd=ROOT, env=_worker_env(tmp_path), text=True, capture_output=True,
    )

    assert result.returncode == 2
    assert "pipeline_root_outside_owner_local_root_or_symlink" in result.stderr


def test_process_wrapper_rejects_pipeline_root_env_mismatch(tmp_path: Path) -> None:
    config_path, env_file = _write_config(tmp_path)
    env_file.write_text(
        env_file.read_text(encoding="utf-8").replace(
            str(tmp_path / ".mango_local" / "pipeline"),
            str(tmp_path / ".mango_local" / "different-pipeline"),
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_file), "process-a-worker"],
        cwd=ROOT, env=_worker_env(tmp_path), text=True, capture_output=True,
    )

    assert result.returncode == 2
    assert "pipeline_root_config_env_mismatch" in result.stderr


@pytest.mark.parametrize("missing_key", ("python_executable", "pipeline_root"))
def test_process_wrapper_reports_missing_required_config_key(
    tmp_path: Path, missing_key: str
) -> None:
    config_path, env_file = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config[missing_key]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_file), "process-a-worker"],
        cwd=ROOT, env=_worker_env(tmp_path), text=True, capture_output=True,
    )

    assert result.returncode == 2
    assert (
        json.loads(result.stderr)["stop_reason"]
        == "config_file_must_be_owner_only_0600_or_invalid"
    )


def test_legacy_local_process_a_keeps_working_before_runtime_relocation(tmp_path: Path) -> None:
    config_path, env_file = _write_config(tmp_path)
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
elif [[ "$1" == *run_mango_calls_pipeline.py ]]; then
  print -r -- '{{"status":"idle","downstream_ready":false}}'
fi
''',
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pipeline_root"] = str(ROOT / "product_data" / "legacy-runtime")
    config["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_file), "process-a"],
        cwd=ROOT, text=True, capture_output=True,
    )

    assert result.returncode == 0
    assert "pipeline_root_outside_owner_local_root_or_symlink" not in result.stderr


def test_config_swap_between_wrapper_and_pipeline_is_rejected(tmp_path: Path) -> None:
    config_path, env_file = _write_config(tmp_path)
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
{shlex.quote(sys.executable)} - "$CONFIG_TO_SWAP" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload.update(require_cutover_authority=True, strict_ready_provenance=True)
path.write_text(json.dumps(payload), encoding="utf-8")
PY
exec {shlex.quote(sys.executable)} "$@"
''',
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pipeline_root"] = str(ROOT / "product_data" / "legacy-runtime")
    config["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_file), "process-a"],
        cwd=ROOT,
        env={**_worker_env(tmp_path), "CONFIG_TO_SWAP": str(config_path)},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert payload["stop_reason"] == "cli_exception:RuntimeError"
    assert "launchctl" not in result.stderr


def test_unchanged_config_matches_wrapper_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from mango_mvp.customer_timeline.calls_two_processes import (
        CallsTwoProcessesConfig,
    )
    from mango_mvp.productization.mango_calls_config import (
        EXPECTED_CONFIG_SHA_ENV,
    )

    config_path, _ = _write_config(tmp_path)
    expected = hashlib.sha256(config_path.read_bytes()).hexdigest()
    monkeypatch.setenv(EXPECTED_CONFIG_SHA_ENV, expected)

    config = CallsTwoProcessesConfig.from_json(config_path)

    assert config.pipeline_root == tmp_path / ".mango_local" / "pipeline"


@pytest.mark.parametrize("command", ["process-a", "process-b", "capture", "pipeline", "watchdog"])
def test_strict_runtime_rejects_unguarded_aliases(
    tmp_path: Path, command: str
) -> None:
    config_path, env_file = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(require_cutover_authority=True, strict_ready_provenance=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_file), command],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert (
        json.loads(result.stderr)["stop_reason"]
        == "strict_runtime_requires_guarded_worker_command"
    )


def test_missing_strict_flags_default_to_strict_runtime(tmp_path: Path) -> None:
    config_path, env_file = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("require_cutover_authority")
    config.pop("strict_ready_provenance")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [str(RUNNER), str(config_path), str(env_file), "process-a"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert (
        json.loads(result.stderr)["stop_reason"]
        == "strict_runtime_requires_guarded_worker_command"
    )


@pytest.mark.parametrize("invalid", ("true", 1, None))
def test_strict_flags_must_be_json_booleans(
    tmp_path: Path, invalid: object
) -> None:
    config_path, env_file = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["require_cutover_authority"] = invalid
    config_path.write_text(json.dumps(config), encoding="utf-8")
    installer = _load_installer()

    with pytest.raises(ValueError, match="JSON boolean"):
        installer.main(
            [
                "--config",
                str(config_path),
                "--env-file",
                str(env_file),
                "--out-dir",
                str(tmp_path / "launchd"),
                "--fast-service",
            ]
        )


def test_strict_flags_must_be_enabled_together(tmp_path: Path) -> None:
    config_path, env_file = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(require_cutover_authority=True, strict_ready_provenance=False)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    installer = _load_installer()

    with pytest.raises(ValueError, match="enabled together"):
        installer.main(
            [
                "--config",
                str(config_path),
                "--env-file",
                str(env_file),
                "--out-dir",
                str(tmp_path / "launchd"),
                "--fast-service",
            ]
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS extended ACL syntax")
@pytest.mark.parametrize("target", ("config", "env"))
def test_runtime_wrapper_rejects_owner_only_file_with_extended_acl(
    tmp_path: Path, target: str
) -> None:
    config_path, env_file = _write_config(tmp_path)
    path = config_path if target == "config" else env_file
    subprocess.run(
        ["/usr/bin/xattr", "-w", "com.mango.synthetic", "1", str(path)],
        check=True,
    )
    subprocess.run(
        ["/bin/chmod", "+a", "everyone allow read", str(path)],
        check=True,
    )
    try:
        result = subprocess.run(
            [str(RUNNER), str(config_path), str(env_file), "process-a"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
    finally:
        subprocess.run(["/bin/chmod", "-N", str(path)], check=True)
        subprocess.run(
            ["/usr/bin/xattr", "-d", "com.mango.synthetic", str(path)],
            check=False,
        )

    assert result.returncode == 2
    expected = (
        "config_file_must_be_owner_only_0600_or_invalid"
        if target == "config"
        else "env_file_must_be_owner_only_0600"
    )
    assert json.loads(result.stderr)["stop_reason"] == expected


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS extended ACL syntax")
def test_installer_rejects_owner_only_config_with_extended_acl(tmp_path: Path) -> None:
    config_path, env_file = _write_config(tmp_path)
    subprocess.run(
        ["/usr/bin/xattr", "-w", "com.mango.synthetic", "1", str(config_path)],
        check=True,
    )
    subprocess.run(
        ["/bin/chmod", "+a", "everyone allow read", str(config_path)],
        check=True,
    )
    installer = _load_installer()
    try:
        with pytest.raises(RuntimeError, match="config file permissions are unsafe"):
            installer.main(
                [
                    "--config",
                    str(config_path),
                    "--env-file",
                    str(env_file),
                    "--out-dir",
                    str(tmp_path / "launchd"),
                    "--fast-service",
                ]
            )
    finally:
        subprocess.run(["/bin/chmod", "-N", str(config_path)], check=True)
        subprocess.run(
            ["/usr/bin/xattr", "-d", "com.mango.synthetic", str(config_path)],
            check=False,
        )


def test_legacy_cycle_entrypoint_is_fail_closed() -> None:
    result = subprocess.run([str(LEGACY_RUNNER)], cwd=ROOT, text=True, capture_output=True)

    assert result.returncode == 2
    assert "deprecated_entrypoint_use_run_mango_calls_process" in result.stderr
    source = LEGACY_RUNNER.read_text(encoding="utf-8")
    assert "source " not in source
    assert "run_mango_calls_pipeline.py" not in source


def test_process_b_pull_rejects_partial_config_before_pipeline(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    clean_repo, head = _clean_git_repo(tmp_path)
    env_path.write_text(env_path.read_text(encoding="utf-8")
                        + f"MANGO_CALLS_EXPECTED_CODE_SHA={head}\n"
                        + "MANGO_CALLS_REMOTE_HOST=main-host\n", encoding="utf-8")
    result = subprocess.run([str(clean_repo / "scripts" / RUNNER.name), str(config_path), str(env_path), "process-b-pull"],
                            cwd=ROOT, env=_worker_env(tmp_path), check=False)
    assert result.returncode == 4


def test_process_b_pull_accepts_remote_drop_before_b_and_never_publishes(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    clean_repo, head = _clean_git_repo(tmp_path)
    captured = tmp_path / "python_args.txt"
    identity, known_hosts = tmp_path / "identity", tmp_path / "known_hosts"
    identity.write_text("identity", encoding="utf-8")
    known_hosts.write_text("known-host", encoding="utf-8")
    identity.chmod(0o600)
    known_hosts.chmod(0o600)
    env_path.write_text(env_path.read_text(encoding="utf-8")
                        + f"MANGO_CALLS_EXPECTED_CODE_SHA={head}\n"
                        + "MANGO_CALLS_REMOTE_HOST=m1-worker\n"
                        + "MANGO_CALLS_REMOTE_DROP_ROOT=/Users/test/.mango_local/drop\n"
                        + f"MANGO_CALLS_REMOTE_INCOMING_ROOT={tmp_path / 'incoming'}\n"
                        + f"MANGO_CALLS_REMOTE_SSH_KEY={identity}\n"
                        + f"MANGO_CALLS_REMOTE_KNOWN_HOSTS={known_hosts}\n", encoding="utf-8")
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
printf -- "--call--\\n" >> "$CAPTURED"
printf "%s\\n" "$@" >> "$CAPTURED"
if [[ "$1" == *pull_mango_calls_drop_remote.py ]]; then
  print -r -- '{{"status":"accepted"}}'
elif [[ "$1" == *run_mango_calls_pipeline.py ]]; then
  print -r -- '{{"status":"idle","stop_reason":"drop_unchanged"}}'
elif [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
fi
''', encoding="utf-8")
    fake_python.chmod(0o700)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["python_executable"] = str(fake_python)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run([str(clean_repo / "scripts" / RUNNER.name), str(config_path), str(env_path), "process-b-pull"], cwd=ROOT,
                            env=_worker_env(tmp_path, CAPTURED=str(captured)), check=False)

    calls = captured.read_text(encoding="utf-8").split("--call--\n")
    pull_index = next(index for index, call in enumerate(calls) if "pull_mango_calls_drop_remote.py" in call)
    assert result.returncode == 0 and pull_index > 0
    assert all("export_daily_mango_calls_resolve.py" not in call for call in calls)


def test_process_b_pull_requires_dedicated_ssh_files(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    clean_repo, head = _clean_git_repo(tmp_path)
    env_path.write_text(
        env_path.read_text(encoding="utf-8")
        + f"MANGO_CALLS_EXPECTED_CODE_SHA={head}\n"
        + "MANGO_CALLS_REMOTE_HOST=m1-worker\n"
        + "MANGO_CALLS_REMOTE_DROP_ROOT=/Users/test/.mango_local/drop\n"
        + f"MANGO_CALLS_REMOTE_INCOMING_ROOT={tmp_path / 'incoming'}\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(clean_repo / "scripts" / RUNNER.name), str(config_path), str(env_path), "process-b-pull"],
        cwd=ROOT, env=_worker_env(tmp_path), text=True, capture_output=True, check=False,
    )

    assert result.returncode == 4
    assert "remote_ssh_files_incomplete" in result.stderr


def test_split_modes_require_exact_clean_code_revision(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    result = subprocess.run([str(RUNNER), str(config_path), str(env_path), "process-a-worker"], cwd=ROOT, env=_worker_env(tmp_path), check=False)
    assert result.returncode == 4
    env_path.write_text(env_path.read_text(encoding="utf-8")
                        + "MANGO_CALLS_EXPECTED_CODE_SHA=0000000000000000000000000000000000000000\n", encoding="utf-8")
    result = subprocess.run([str(RUNNER), str(config_path), str(env_path), "process-a-worker"], cwd=ROOT, env=_worker_env(tmp_path), check=False)
    assert result.returncode == 4


def test_split_runner_ignores_inherited_git_repository_redirection(
    tmp_path: Path,
) -> None:
    config_path, env_path = _write_config(tmp_path)
    clean_repo, head = _clean_git_repo(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q"], cwd=foreign, check=True)
    env_path.write_text(
        env_path.read_text(encoding="utf-8")
        + f"MANGO_CALLS_EXPECTED_CODE_SHA={head}\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            str(clean_repo / "scripts" / RUNNER.name),
            str(config_path),
            str(env_path),
            "process-a-worker",
        ],
        cwd=ROOT,
        env=_worker_env(
            tmp_path,
            GIT_DIR=str(foreign / ".git"),
            GIT_WORK_TREE=str(foreign),
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 4
    assert "split_code_revision_mismatch_or_dirty" not in result.stderr


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_split_runner_rejects_hidden_index_flags(
    tmp_path: Path, flag: str
) -> None:
    config_path, env_path = _write_config(tmp_path)
    clean_repo, head = _clean_git_repo(tmp_path)
    relative_runner = f"scripts/{RUNNER.name}"
    subprocess.run(
        ["/usr/bin/git", "update-index", flag, relative_runner],
        cwd=clean_repo,
        check=True,
    )
    runner = clean_repo / relative_runner
    runner.write_text(
        runner.read_text(encoding="utf-8") + "\n# hidden dirty fixture\n",
        encoding="utf-8",
    )
    env_path.write_text(
        env_path.read_text(encoding="utf-8")
        + f"MANGO_CALLS_EXPECTED_CODE_SHA={head}\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(runner), str(config_path), str(env_path), "process-a-worker"],
        cwd=ROOT,
        env=_worker_env(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 4
    assert "split_code_revision_mismatch_or_dirty" in result.stderr


def test_process_a_partial_ready_starts_b_and_preserves_rc_one(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    capture = tmp_path / "launchctl_args.txt"
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
else
  print -r -- '{{"status":"partial","downstream_ready":true}}'
  exit 1
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
        '#!/bin/zsh\nprintf "%s\\n" "$@" > "$CAPTURED_LAUNCHCTL"\n', encoding="utf-8"
    )
    fake_launchctl.chmod(0o700)
    runner = _copy_runner(tmp_path, fake_launchctl)

    result = subprocess.run(
        [str(runner), str(config_path), str(env_path), "process-a"],
        cwd=ROOT,
        env={**__import__("os").environ, "CAPTURED_LAUNCHCTL": str(capture)},
        check=False,
    )

    assert result.returncode == 1
    assert capture.read_text(encoding="utf-8").splitlines()[0] == "kickstart"


def test_process_a_reports_kickstart_failure(tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    fake_python = tmp_path / "configured-python"
    fake_python.write_text(
        f'''#!/bin/zsh
if [[ "$1" == "-c" ]]; then
  {shlex.quote(sys.executable)} "$@"
else
  print -r -- '{{"status":"ok","downstream_ready":true}}'
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
    runner = _copy_runner(tmp_path, fake_launchctl)

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
    _allow_test_install_path(monkeypatch, installer, out_dir)
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


@pytest.mark.parametrize("mode", [(), ("--process-a-only",), ("--fast-service",)])
def test_every_process_a_install_requires_cutover_authority_before_launchctl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: tuple[str, ...],
) -> None:
    config_path, env_path = _write_config(tmp_path)
    out_dir = tmp_path / "launchd"
    installer = _load_installer()
    monkeypatch.setattr(
        installer,
        "_standard_plist",
        lambda label: out_dir.resolve() / f"{label}.plist",
    )
    launchctl_called = False

    def reject_authority(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic_authority_rejected")

    def forbidden_launchctl(*_args: object, **_kwargs: object) -> object:
        nonlocal launchctl_called
        launchctl_called = True
        return subprocess.CompletedProcess([], 1)

    monkeypatch.setattr(
        installer, "_validate_process_a_install_authority", reject_authority
    )
    monkeypatch.setattr(installer.subprocess, "run", forbidden_launchctl)

    with pytest.raises(RuntimeError, match="synthetic_authority_rejected"):
        installer.main(
            [
                "--config",
                str(config_path),
                "--env-file",
                str(env_path),
                "--out-dir",
                str(out_dir),
                *mode,
                "--install",
            ]
        )

    assert launchctl_called is False


@pytest.mark.parametrize("mode", [(), ("--process-a-only",)])
@pytest.mark.parametrize(
    "conflict_kind", ["loaded_capture", "pipeline_plist", "pipeline_lock"]
)
def test_monolithic_process_a_install_rejects_existing_fast_topology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: tuple[str, ...],
    conflict_kind: str,
) -> None:
    config_path, env_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = tmp_path / "launchd"
    installer = _load_installer()
    _allow_test_install_path(monkeypatch, installer, out_dir)
    out_dir.mkdir()

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        loaded = (
            conflict_kind == "loaded_capture"
            and command[:2] == ["launchctl", "print"]
            and command[2].endswith(f"/{installer.LABEL_CAPTURE}")
        )
        return subprocess.CompletedProcess(command, 0 if loaded else 1)

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    lock_handle = None
    if conflict_kind == "pipeline_plist":
        (out_dir / f"{installer.LABEL_PIPELINE}.plist").write_text(
            "synthetic", encoding="utf-8"
        )
    elif conflict_kind == "pipeline_lock":
        lock_path = Path(config["pipeline_root"]) / "locks" / "pipeline.lock"
        lock_path.parent.mkdir(parents=True)
        lock_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        lock_handle = lock_path.open("r+", encoding="utf-8")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    try:
        with pytest.raises(RuntimeError, match="existing fast-service topology"):
            installer.main(
                [
                    "--config",
                    str(config_path),
                    "--env-file",
                    str(env_path),
                    "--out-dir",
                    str(out_dir),
                    *mode,
                    "--install",
                ]
            )
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

    assert not (out_dir / f"{installer.LABEL_A}.plist").exists()


def test_process_a_only_install_refuses_if_process_b_is_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, env_path = _write_config(tmp_path)
    installer = _load_installer()
    _allow_test_install_path(monkeypatch, installer, tmp_path / "launchd")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        loaded_b = command[:2] == ["launchctl", "print"] and command[2].endswith(installer.LABEL_B)
        return subprocess.CompletedProcess(command, 0 if loaded_b else 1, stdout="", stderr="")

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Process B or legacy service is active"):
        installer.main([
            "--config", str(config_path), "--env-file", str(env_path),
            "--out-dir", str(tmp_path / "launchd"), "--process-a-only", "--install",
        ])


@pytest.mark.parametrize("single_flag", ["--process-a-only", "--process-b-only"])
def test_single_role_install_refuses_loaded_legacy_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, single_flag: str
) -> None:
    config_path, env_path = _write_config(tmp_path)
    installer = _load_installer()
    _allow_test_install_path(monkeypatch, installer, tmp_path / "launchd")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        legacy = command[:2] == ["launchctl", "print"] and command[2].endswith(installer.OLD_LABEL)
        return subprocess.CompletedProcess(command, 0 if legacy else 1, stdout="", stderr="")

    monkeypatch.setattr(installer.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="legacy service is active"):
        installer.main([
            "--config", str(config_path), "--env-file", str(env_path), "--out-dir", str(tmp_path / "launchd"),
            single_flag, "--install",
        ])


@pytest.mark.parametrize(
    "single_flag,lock_name,error",
    [
        ("--process-b-only", "process_a", "Process A or legacy service is active"),
        ("--process-a-only", "process_b", "Process B or legacy service is active"),
    ],
)
def test_single_role_install_refuses_opposite_live_process_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, single_flag: str, lock_name: str, error: str
) -> None:
    config_path, env_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    locks = Path(config["pipeline_root"]) / "locks"
    locks.mkdir(parents=True)
    lock_path = locks / f"{lock_name}.lock"
    lock_path.write_text(json.dumps({"pid": __import__("os").getpid()}), encoding="utf-8")
    lock_handle = lock_path.open("r+", encoding="utf-8")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    installer = _load_installer()
    _allow_test_install_path(monkeypatch, installer, tmp_path / "launchd")
    monkeypatch.setattr(installer.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1))

    try:
        with pytest.raises(RuntimeError, match=error):
            installer.main([
                "--config", str(config_path), "--env-file", str(env_path), "--out-dir", str(tmp_path / "launchd"),
                single_flag, "--install",
            ])
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def test_single_role_install_uses_flock_and_standard_launchagents_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, env_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    locks = Path(config["pipeline_root"]) / "locks"
    locks.mkdir(parents=True)
    (locks / "process_a.lock").write_text(json.dumps({"pid": __import__("os").getpid()}), encoding="utf-8")
    installer = _load_installer()
    monkeypatch.setattr(installer.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1))
    monkeypatch.setattr(installer, "_conflicting_plist_exists", lambda labels: False)

    assert installer._live_process_lock(Path(config["pipeline_root"]), "process_a") is False
    with pytest.raises(RuntimeError, match="standard LaunchAgents path"):
        installer.main([
            "--config", str(config_path), "--env-file", str(env_path), "--out-dir", str(tmp_path / "preview"),
            "--process-b-only", "--install",
        ])


def test_single_role_install_detects_conflicting_plist_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    installer = _load_installer()
    conflict = tmp_path / "legacy.plist"
    conflict.write_text("legacy", encoding="utf-8")
    monkeypatch.setattr(installer, "_standard_plist", lambda label: conflict)
    assert installer._conflicting_plist_exists((installer.OLD_LABEL,)) is True


def test_partial_install_rolls_back_process_a(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path, env_path = _write_config(tmp_path)
    out_dir = tmp_path / "launchd"
    installer = _load_installer()
    _allow_test_install_path(monkeypatch, installer, out_dir)
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
    _allow_test_install_path(monkeypatch, installer, out_dir)
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
    _allow_test_install_path(monkeypatch, installer, out_dir)
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
