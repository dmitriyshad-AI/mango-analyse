#!/usr/bin/env python3
"""M1 readiness probe: opt-in synthetic ASR, never real audio or external writes."""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import pwd
import shutil
import subprocess
import sys
import tempfile
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mango_calls_service_contract import (  # noqa: E402
    approved_runtime_fingerprint,
    current_git_sha,
    git_worktree_is_clean,
    load_owner_only_json,
    read_host_id,
    read_stable_regular_bytes,
    stage_capacity_report,
    validate_runtime_fingerprint,
)
from mango_mvp.productization.owner_only_io import (  # noqa: E402
    path_has_cloud_marker,
    validate_owner_only_directory,
)
from mango_mvp.customer_timeline.calls_two_processes import (  # noqa: E402
    CallsTwoProcessesConfig,
    codex_network_available,
    command_path,
    cutover_authority_report,
    ensure_codex_runtime_anchor,
    observe_runtime_fingerprint,
    prepare_codex_home,
)


CODEX_HOME_ALLOWLIST = {
    "auth.json",
    "installation_id",
    "models_cache.json",
    "config.toml",
    "AGENTS.md",
}
CODEX_RUNTIME_HOME_ALLOWLIST = {
    "worker",
    "transcribe",
    "backfill_second_asr",
    "resolve",
    "analyze",
}

CAPACITY_CHECKS = frozenset(
    {
        "capacity_2x_and_memory_ok",
        "measurement_evidence_bound_and_fresh",
    }
)


def command_output(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            list(command), text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def physical_memory_bytes() -> int:
    raw = command_output(("sysctl", "-n", "hw.memsize"))
    try:
        return int(raw)
    except ValueError:
        return 0


def inspect_codex_home(path: Path) -> Mapping[str, Any]:
    candidate = path.expanduser()
    try:
        resolved = validate_owner_only_directory(
            candidate,
            label="probe_codex_home",
            owner_only_mode=0o700,
        )
        root_private = True
    except RuntimeError:
        resolved = candidate.resolve(strict=False)
        root_private = False
    cloud = path_has_cloud_marker(resolved)
    under_repo = resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents
    under_owner_local = (Path.home() / ".mango_local").resolve(strict=False) in resolved.parents
    unknown: list[str] = []
    unsafe_modes: list[str] = []
    if candidate.is_dir() and not candidate.is_symlink():
        for runtime_home in candidate.iterdir():
            if runtime_home.name in CODEX_HOME_ALLOWLIST:
                try:
                    read_stable_regular_bytes(
                        runtime_home,
                        label=(
                            "probe_codex_"
                            f"{runtime_home.name.replace('.', '_')}"
                        ),
                        owner_only_mode=0o600,
                    )
                except (OSError, RuntimeError):
                    unsafe_modes.append(runtime_home.name)
                continue
            if runtime_home.name not in CODEX_RUNTIME_HOME_ALLOWLIST:
                unknown.append(runtime_home.name)
                continue
            try:
                validate_owner_only_directory(
                    runtime_home,
                    label=f"probe_codex_{runtime_home.name}",
                    owner_only_mode=0o700,
                )
            except RuntimeError:
                unsafe_modes.append(runtime_home.name)
                continue
            for entry in runtime_home.iterdir():
                relative = f"{runtime_home.name}/{entry.name}"
                if entry.name not in CODEX_HOME_ALLOWLIST:
                    unknown.append(relative)
                    continue
                try:
                    read_stable_regular_bytes(
                        entry,
                        label=(
                            "probe_codex_"
                            f"{runtime_home.name}_{entry.name.replace('.', '_')}"
                        ),
                        owner_only_mode=0o600,
                    )
                except (OSError, RuntimeError):
                    unsafe_modes.append(relative)
    ok = bool(
        root_private
        and under_owner_local
        and not cloud
        and not under_repo
        and not unknown
        and not unsafe_modes
    )
    return {
        "ok": ok,
        "path": str(resolved),
        "owner_only_0700": root_private,
        "under_owner_local": under_owner_local,
        "outside_cloud": not cloud,
        "outside_repo": not under_repo,
        "unknown_files": sorted(unknown),
        "unsafe_files": sorted(unsafe_modes),
        "persistent_session_or_history": any(
            Path(name).name.casefold()
            in {"sessions", "history.jsonl", "session.json"}
            for name in unknown
        ),
    }


def _owner_env(path: Path) -> Mapping[str, str]:
    try:
        raw = read_stable_regular_bytes(
            path, label="probe_env", owner_only_mode=0o600
        ).decode("utf-8")
    except (RuntimeError, UnicodeDecodeError) as exc:
        raise RuntimeError("probe env must be owner-only 0600") from exc
    from dotenv import dotenv_values

    return {
        str(key): str(value)
        for key, value in dotenv_values(stream=io.StringIO(raw)).items()
        if key and value is not None
    }


def probe_mango_readonly(config: Mapping[str, Any]) -> bool:
    from mango_mvp.productization.mango_office_client import (
        MangoOfficeClient,
        MangoOfficeCredentials,
    )

    env = _owner_env(Path(str(config.get("mango_env_path") or "")).expanduser())
    end = datetime.now(timezone.utc) - timedelta(days=60)
    start = end - timedelta(minutes=1)
    client = MangoOfficeClient(
        MangoOfficeCredentials(
            str(env.get("MANGO_OFFICE_API_KEY") or ""),
            str(env.get("MANGO_OFFICE_API_SALT") or ""),
        ),
        base_url=str(env.get("MANGO_OFFICE_BASE_URL") or config.get("base_url") or ""),
        timeout_sec=30,
    )
    rows = client.poll_call_history(since=start, until=end)
    return isinstance(rows, (list, tuple))


def probe_tallanto_readonly(config: Mapping[str, Any]) -> bool:
    from mango_mvp.amocrm_runtime.tallanto_api import TallantoApiClient, TallantoApiConfig

    env = _owner_env(Path(str(config.get("tallanto_env_path") or "")).expanduser())
    client = TallantoApiClient(
        TallantoApiConfig(
            base_url=str(env.get("CRM_TALLANTO_BASE_URL") or ""),
            api_token=str(env.get("CRM_TALLANTO_API_TOKEN") or ""),
            rest_path=str(env.get("CRM_TALLANTO_STUDENT_PATH") or "/service/api/rest.php"),
        )
    )
    payload = client.get_entry_list(
        module="Contact",
        select_fields=("id", "date_modified"),
        order_by="date_modified DESC",
        offset=0,
    )
    return isinstance(payload, Mapping) and isinstance(payload.get("entry_list"), list)


def probe_google_readonly(config: Mapping[str, Any]) -> Mapping[str, bool]:
    from scripts.publish_current_mango_calls_google import (
        GoogleGateway,
        authorized_session,
        exact_acl_ok,
        load_google_config,
    )

    google_config = load_google_config(
        Path(str(config.get("google_current_config_path") or "")).expanduser()
    )
    gateway = GoogleGateway(
        authorized_session(google_config["credentials"]),
        str(google_config["spreadsheet_id"]),
    )
    acl_ok = exact_acl_ok(
        gateway.permissions(),
        owner_email=str(google_config["owner_email"]),
        allowed_emails=google_config["allowed_emails"],
        expected_roles=google_config["expected_roles"],
    )
    metadata_ok = isinstance(gateway.sheets(), Sequence)
    return {
        "google_spreadsheet_acl_ok": acl_ok,
        "google_metadata_readback_ok": metadata_ok,
    }


def probe_yandex_marker(config: Mapping[str, Any]) -> bool:
    marker_path = Path(str(config.get("yandex_private_marker_path") or "")).expanduser()
    marker = load_owner_only_json(marker_path, label="yandex_private_marker")
    folder = Path(str(marker.get("folder") or "")).expanduser()
    return bool(
        marker.get("schema_version") == "mango_yandex_private_folder_v1"
        and marker.get("private_acl_readback_ok") is True
        and folder.is_dir()
        and not folder.is_symlink()
        and marker_path.parent.resolve() == folder.resolve()
    )


def probe_time_sync() -> bool:
    output = command_output(("systemsetup", "-getusingnetworktime")).casefold()
    return "on" in output or "вкл" in output


def machine_timezone() -> str:
    """Return the canonical IANA timezone configured for the host."""
    try:
        target = os.readlink("/etc/localtime")
    except OSError:
        target = ""
    marker = "/zoneinfo/"
    if marker in target:
        return target.split(marker, 1)[1].strip()
    output = command_output(("/usr/sbin/systemsetup", "-gettimezone"))
    if ":" in output:
        return output.split(":", 1)[1].strip()
    return ""


def probe_conflicting_launchd() -> bool:
    labels = (
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
    )
    domain = f"gui/{os.getuid()}"
    return all(
        subprocess.run(
            ["launchctl", "print", f"{domain}/{label}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        != 0
        for label in labels
    )


def probe_offline_models(config: CallsTwoProcessesConfig) -> Mapping[str, bool]:
    result = {
        "offline_whisper_synthetic_attempted": False,
        "offline_gigaam_synthetic_attempted": False,
        "offline_whisper_synthetic_completed": False,
        "offline_gigaam_synthetic_completed": False,
        "offline_whisper_synthetic_ok": False,
        "offline_gigaam_synthetic_ok": False,
    }
    with tempfile.TemporaryDirectory(prefix="mango_calls_probe_") as raw_dir:
        sample = Path(raw_dir) / "synthetic.wav"
        with wave.open(str(sample), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(b"\0\0" * 16_000)
        mlx_code = (
            "import mlx_whisper,sys;"
            "mlx_whisper.transcribe(sys.argv[1],path_or_hf_repo=sys.argv[2],"
            "language='ru',condition_on_previous_text=False,word_timestamps=True)"
        )
        try:
            result["offline_whisper_synthetic_attempted"] = True
            mlx = subprocess.run(
                [
                    str(config.python_executable),
                    "-c",
                    mlx_code,
                    str(sample),
                    str(config.mlx_whisper_snapshot_path or ""),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired):
            return result
        result["offline_whisper_synthetic_completed"] = True
        result["offline_whisper_synthetic_ok"] = mlx.returncode == 0
        # Sequential by construction: GigaAM starts only after Whisper exits.
        gigaam_code = (
            "import gigaam,sys;"
            "m=gigaam.load_model('v2_rnnt',device='cpu',fp16_encoder=False);"
            "m.transcribe(sys.argv[1])"
        )
        try:
            result["offline_gigaam_synthetic_attempted"] = True
            gigaam = subprocess.run(
                [str(config.python_executable), "-c", gigaam_code, str(sample)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired):
            return result
        result["offline_gigaam_synthetic_completed"] = True
        result["offline_gigaam_synthetic_ok"] = gigaam.returncode == 0
    return result


def probe_codex_models(config: CallsTwoProcessesConfig) -> Mapping[str, bool]:
    """Run content-free access probes for the exact Resolve/Analyze models."""

    result = {
        "codex_auth_probe_attempted": False,
        "codex_resolve_model_access_attempted": False,
        "codex_analyze_model_access_attempted": False,
        "codex_resolve_model_access_completed": False,
        "codex_analyze_model_access_completed": False,
        "codex_authenticated_ok": False,
        "codex_network_ok": False,
        "codex_resolve_model_access_ok": False,
        "codex_analyze_model_access_ok": False,
    }
    if not (
        config.codex_binary.is_file()
        and os.access(config.codex_binary, os.X_OK)
    ):
        return result
    wrapper = ROOT / "scripts" / "run_codex_cli_isolated.sh"
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        return result
    result["codex_network_ok"] = codex_network_available()
    try:
        probe_parent = ensure_codex_runtime_anchor(config)
        with tempfile.TemporaryDirectory(
            prefix=".mango-codex-model-probe-",
            dir=probe_parent,
        ) as raw_probe_root:
            probe_root = Path(raw_probe_root)
            probe_root.chmod(0o700)
            process_home = probe_root / "home"
            process_tmp = probe_root / "tmp"
            process_home.mkdir(mode=0o700)
            process_tmp.mkdir(mode=0o700)

            def isolated_env(
                codex_home: Path, *, include_real_binary: bool
            ) -> dict[str, str]:
                environment = {
                    "HOME": str(process_home),
                    "CODEX_HOME": str(codex_home),
                    "PATH": command_path(config),
                    "TMPDIR": str(process_tmp),
                    "LANG": "en_US.UTF-8",
                    "LC_ALL": "en_US.UTF-8",
                    "NO_COLOR": "1",
                }
                if include_real_binary:
                    environment["MANGO_CODEX_REAL_BIN"] = str(
                        config.codex_binary
                    )
                    environment["MANGO_CODEX_PROCESS_HOME"] = str(
                        process_home
                    )
                    environment["MANGO_CODEX_PROCESS_TMPDIR"] = str(
                        process_tmp
                    )
                return environment

            auth_home = prepare_codex_home(
                probe_root / "probe-auth",
                strict=config.strict_ready_provenance,
            )
            result["codex_auth_probe_attempted"] = True
            auth = subprocess.run(
                [str(config.codex_binary), "login", "status"],
                cwd=process_home,
                env=isolated_env(auth_home, include_real_binary=False),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=30,
            )
            result["codex_authenticated_ok"] = auth.returncode == 0
            if not (
                result["codex_authenticated_ok"]
                and result["codex_network_ok"]
            ):
                return result

            prompt = (
                "Synthetic access probe only. Do not inspect files or call "
                "tools. Reply with exactly OK."
            )
            for label, model in (
                ("resolve", config.codex_resolve_model),
                ("analyze", config.codex_analyze_model),
            ):
                codex_home = prepare_codex_home(
                    probe_root / f"probe-{label}",
                    strict=config.strict_ready_provenance,
                )
                with tempfile.NamedTemporaryFile(
                    prefix=f"mango_{label}_access_",
                    suffix=".txt",
                    dir=process_tmp,
                ) as output_file:
                    command = [
                        str(wrapper),
                        "exec",
                        "--skip-git-repo-check",
                    ]
                    if label == "analyze":
                        command.append("--ignore-user-config")
                    command.extend(
                        [
                            "--sandbox",
                            "read-only",
                            "--model",
                            model,
                            "--output-last-message",
                            output_file.name,
                            "-c",
                            f'service_tier="{config.codex_service_tier}"',
                        ]
                    )
                    effort = config.codex_reasoning_effort.strip().lower()
                    if effort in {"low", "medium", "high"}:
                        command.extend(
                            ["-c", f'model_reasoning_effort="{effort}"']
                        )
                    command.append("-" if label == "analyze" else prompt)
                    result[f"codex_{label}_model_access_attempted"] = True
                    completed = subprocess.run(
                        command,
                        cwd=process_home,
                        input=prompt if label == "analyze" else None,
                        env=isolated_env(
                            codex_home,
                            include_real_binary=True,
                        ),
                        text=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=180,
                    )
                    result[f"codex_{label}_model_access_completed"] = True
                    model_reply = Path(output_file.name).read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).strip()
                    result[f"codex_{label}_model_access_ok"] = bool(
                        completed.returncode == 0 and model_reply == "OK"
                    )
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return result
    return result


def _check(action: Any) -> bool:
    try:
        return bool(action())
    except Exception:
        return False


def _measurement_evidence_ok(
    evidence: Mapping[str, Any], *, expected_sha: str, host_id: str
) -> bool:
    if evidence.get("schema_version") != "m1_mango_calls_measurements_v1":
        return False
    if evidence.get("expected_code_sha") != expected_sha or evidence.get("host_id") != host_id:
        return False
    try:
        captured = datetime.fromisoformat(
            str(evidence.get("captured_at_utc") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return False
    current = datetime.now(timezone.utc)
    if (
        captured.tzinfo is None
        or captured.astimezone(timezone.utc) > current
        or current - captured.astimezone(timezone.utc) > timedelta(days=7)
    ):
        return False
    return True


def probe(
    config: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    parsed_config: Optional[CallsTwoProcessesConfig] = None,
    run_offline_model_probes: bool = False,
    run_codex_model_probes: bool = False,
) -> Mapping[str, Any]:
    pipeline_root = Path(str(config.get("pipeline_root") or "")).expanduser()
    codex_home = Path(str(config.get("codex_home_root") or "")).expanduser()
    expected_sha = str(config.get("expected_code_sha") or "")
    actual_sha = current_git_sha(ROOT)
    dirty = not git_worktree_is_clean(ROOT)
    memory = physical_memory_bytes()
    disk = shutil.disk_usage(pipeline_root.parent if pipeline_root.parent.exists() else Path.home())
    runtime_observation = (
        observe_runtime_fingerprint(parsed_config)
        if parsed_config is not None
        else {"ok": False, "fingerprint": {}, "errors": ["config_not_parsed"]}
    )
    actual_fingerprint = runtime_observation.get("fingerprint")
    fingerprint_errors = validate_runtime_fingerprint(actual_fingerprint)
    google = {"google_spreadsheet_acl_ok": False, "google_metadata_readback_ok": False}
    try:
        google = dict(probe_google_readonly(config))
    except Exception:
        pass
    offline = {
        "offline_whisper_synthetic_attempted": False,
        "offline_gigaam_synthetic_attempted": False,
        "offline_whisper_synthetic_completed": False,
        "offline_gigaam_synthetic_completed": False,
        "offline_whisper_synthetic_ok": False,
        "offline_gigaam_synthetic_ok": False,
    }
    if run_offline_model_probes and parsed_config is not None:
        try:
            offline = dict(probe_offline_models(parsed_config))
        except Exception:
            pass
    codex_models = {
        "codex_auth_probe_attempted": False,
        "codex_resolve_model_access_attempted": False,
        "codex_analyze_model_access_attempted": False,
        "codex_resolve_model_access_completed": False,
        "codex_analyze_model_access_completed": False,
        "codex_authenticated_ok": False,
        "codex_network_ok": False,
        "codex_resolve_model_access_ok": False,
        "codex_analyze_model_access_ok": False,
    }
    if run_codex_model_probes and parsed_config is not None:
        try:
            codex_models = dict(probe_codex_models(parsed_config))
        except Exception:
            pass
    if parsed_config is not None:
        try:
            ensure_codex_runtime_anchor(parsed_config)
        except Exception:
            pass
    codex = inspect_codex_home(codex_home)
    wrapper = ROOT / "scripts" / "run_mango_calls_process.sh"
    wrapper_text = wrapper.read_text(encoding="utf-8", errors="ignore")
    access = {
        "mango_readonly_ok": _check(lambda: probe_mango_readonly(config)),
        "tallanto_readonly_ok": _check(lambda: probe_tallanto_readonly(config)),
        **google,
        "yandex_private_marker_ok": _check(lambda: probe_yandex_marker(config)),
        **offline,
        "time_sync_ok": probe_time_sync(),
        "sleep_prevented_during_processing": bool(
            Path("/usr/bin/caffeinate").is_file() and "caffeinate" in wrapper_text
        ),
        "conflicting_launchd_absent": probe_conflicting_launchd(),
        **codex_models,
    }
    host_id = ""
    expected_active_host_id = str(
        config.get("expected_active_host_id") or ""
    ).strip()
    authority: Mapping[str, Any] = {
        "ok": False,
        "errors": ["cutover_authority_not_checked"],
    }
    if parsed_config is not None:
        try:
            host_id = read_host_id(parsed_config.host_id_file)
            authority = cutover_authority_report(
                parsed_config,
                initialize_lineage=False,
            )
        except (OSError, RuntimeError):
            pass
    host_identity_ok = bool(
        host_id
        and expected_active_host_id == host_id
        and getattr(parsed_config, "expected_active_host_id", None) == host_id
        and getattr(parsed_config, "strict_ready_provenance", False)
        and authority.get("ok") is True
        and authority.get("active_host_id") == host_id
        and getattr(parsed_config, "expected_previous_host_id", None)
        != host_id
    )
    measurements_ok = _measurement_evidence_ok(
        evidence, expected_sha=expected_sha, host_id=host_id
    )
    configured_timezone = machine_timezone()
    try:
        runtime_user = pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError):
        runtime_user = ""
    benchmark = evidence.get("controlled_10") if measurements_ok else None
    peak_snapshot = evidence.get("mango_peak_60d") if measurements_ok else None
    capacity = (
        stage_capacity_report(
            benchmark=benchmark,
            peak_snapshot=peak_snapshot,
            physical_memory_bytes=memory,
        )
        if isinstance(benchmark, Mapping) and isinstance(peak_snapshot, Mapping)
        else {"status": "not_measured", "capacity_ok": False}
    )
    checks = {
        "darwin_arm64": platform.system() == "Darwin" and platform.machine() == "arm64",
        "physical_memory_at_least_32_gib": memory >= 32 * 1024**3,
        "free_disk_at_least_40_gib": disk.free >= 40 * 1024**3,
        "expected_clean_git_sha": bool(
            len(expected_sha) == 40 and actual_sha == expected_sha and not dirty
        ),
        "active_m1_host_identity_bound": host_identity_ok,
        "runtime_user_is_dmitriy": runtime_user == "dmitriy",
        "runtime_fingerprint_approved": runtime_observation.get("ok") is True and not fingerprint_errors,
        "isolated_codex_home_ok": codex["ok"] is True,
        "capacity_2x_and_memory_ok": capacity.get("capacity_ok") is True,
        "measurement_evidence_bound_and_fresh": measurements_ok,
        "timezone_is_europe_moscow": configured_timezone == "Europe/Moscow",
        "ffmpeg_present": shutil.which("ffmpeg") is not None,
        "ffprobe_present": shutil.which("ffprobe") is not None,
        **access,
    }
    controlled_1_checks = {
        key: value for key, value in checks.items() if key not in CAPACITY_CHECKS
    }
    capacity_checks = {key: checks[key] for key in sorted(CAPACITY_CHECKS)}
    controlled_1_ok = all(controlled_1_checks.values())
    service_capacity_ok = all(capacity_checks.values())
    production_service_ok = controlled_1_ok and service_capacity_ok
    asr_attempted = bool(
        offline.get("offline_whisper_synthetic_attempted")
        or offline.get("offline_gigaam_synthetic_attempted")
    )
    asr_completed = bool(
        offline.get("offline_whisper_synthetic_completed")
        or offline.get("offline_gigaam_synthetic_completed")
    )
    codex_attempted = bool(
        codex_models.get("codex_resolve_model_access_attempted")
        or codex_models.get("codex_analyze_model_access_attempted")
    )
    codex_completed = bool(
        codex_models.get("codex_resolve_model_access_completed")
        or codex_models.get("codex_analyze_model_access_completed")
    )
    codex_success = bool(
        codex_models.get("codex_resolve_model_access_ok")
        or codex_models.get("codex_analyze_model_access_ok")
    )
    confirmed_external_model_request: Optional[bool] = (
        True if codex_success else None if codex_attempted else False
    )
    return {
        "schema_version": "m1_mango_calls_access_probe_v1",
        "mode": (
            "read_only_access_plus_synthetic_offline_and_codex_models"
            if run_offline_model_probes and run_codex_model_probes
            else "read_only_access_plus_synthetic_offline_models"
            if run_offline_model_probes
            else "read_only_access_plus_synthetic_codex_models"
            if run_codex_model_probes
            else "read_only_access_no_models_invoked"
        ),
        # Backward-compatible alias: the historical host verdict remains the
        # stricter service/capacity verdict.  A one-call pilot can now be
        # assessed independently without pretending controlled-10 exists.
        "host_readiness": "OK" if production_service_ok else "STOP",
        "controlled_1_readiness": {
            "status": "OK" if controlled_1_ok else "STOP",
            "ok": controlled_1_ok,
            "checks": controlled_1_checks,
            "requires_controlled_10": False,
        },
        "service_capacity_readiness": {
            "status": "OK" if service_capacity_ok else "STOP",
            "ok": service_capacity_ok,
            "checks": capacity_checks,
            "requires_controlled_10": True,
        },
        "production_service_readiness": {
            "status": "OK" if production_service_ok else "STOP",
            "ok": production_service_ok,
            "requires_controlled_1": True,
            "requires_service_capacity": True,
        },
        "checks": checks,
        "runtime_fingerprint_errors": fingerprint_errors,
        "runtime_observation_errors": runtime_observation.get("errors"),
        "codex_home": codex,
        "host_identity": {
            "ok": host_identity_ok,
            "actual_host_id": host_id,
            "expected_active_host_id": expected_active_host_id,
            "cutover_authority_ok": authority.get("ok") is True,
        },
        "capacity": capacity,
        "machine": {
            "model": command_output(("sysctl", "-n", "hw.model")),
            "physical_memory_bytes": memory,
            "free_disk_bytes": disk.free,
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "timezone": configured_timezone,
            "user": runtime_user,
        },
        "approved_runtime_fingerprint": approved_runtime_fingerprint(),
        "offline_model_probes_requested": run_offline_model_probes,
        "codex_model_access_probes_requested": run_codex_model_probes,
        "synthetic_asr_attempted": asr_attempted,
        "synthetic_asr_completed": asr_completed,
        "runs_asr": asr_completed,
        "synthetic_asr_executed": asr_completed,
        "runs_real_asr": False,
        "codex_model_access_probe_attempted": codex_attempted,
        "codex_model_access_probe_completed": codex_completed,
        "runs_codex_model_access_probes": codex_completed,
        "runs_content_free_network_model_inference": (
            confirmed_external_model_request
        ),
        "external_model_requests_made": confirmed_external_model_request,
        "resolve_service_executed": False,
        "analyze_service_executed": False,
        "runs_resolve_analyze_pipeline": False,
        "runs_real_audio": False,
        "real_audio_processed": False,
        "runs_resolve_analyze": False,
        "uses_customer_content": False,
        "customer_content_used": False,
        "writes_external_systems": False,
        "external_system_writes": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only M1 Calls access and machine probe.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        help="Owner-only JSON of separately collected read-only/synthetic evidence.",
    )
    parser.add_argument(
        "--run-offline-model-probes",
        action="store_true",
        help="Run the approved synthetic Whisper probe, then GigaAM; never real audio.",
    )
    parser.add_argument(
        "--run-codex-model-probes",
        action="store_true",
        help="Run content-free access probes for the exact Resolve and Analyze models.",
    )
    parser.add_argument(
        "--readiness-target",
        choices=("service", "controlled-1", "capacity"),
        default="service",
        help="Select the verdict used as the process exit status.",
    )
    args = parser.parse_args(argv)
    config = load_owner_only_json(args.config, label="probe_config")
    parsed_config = CallsTwoProcessesConfig.from_json(args.config)
    evidence = (
        load_owner_only_json(args.evidence, label="probe_evidence")
        if args.evidence
        else {}
    )
    report = probe(
        config,
        evidence,
        parsed_config=parsed_config,
        run_offline_model_probes=args.run_offline_model_probes,
        run_codex_model_probes=args.run_codex_model_probes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    selected = (
        report["controlled_1_readiness"]
        if args.readiness_target == "controlled-1"
        else report["service_capacity_readiness"]
        if args.readiness_target == "capacity"
        else report["production_service_readiness"]
    )
    return 0 if selected["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
