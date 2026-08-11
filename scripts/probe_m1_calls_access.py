#!/usr/bin/env python3
"""Read-only M1 Calls readiness probe; it never invokes ASR or external writes."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
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
    stage_capacity_report,
    validate_runtime_fingerprint,
)
from mango_mvp.customer_timeline.calls_two_processes import (  # noqa: E402
    CallsTwoProcessesConfig,
    observe_runtime_fingerprint,
)


CODEX_HOME_ALLOWLIST = {
    "auth.json",
    "installation_id",
    "models_cache.json",
    "config.toml",
    "AGENTS.md",
}


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
    resolved = path.expanduser().resolve(strict=False)
    cloud = any(
        marker in part.casefold()
        for part in resolved.parts
        for marker in ("yandex.disk", "icloud", "mobile documents", "dropbox", "onedrive")
    )
    under_repo = resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents
    under_owner_local = (Path.home() / ".mango_local").resolve(strict=False) in resolved.parents
    unknown: list[str] = []
    unsafe_modes: list[str] = []
    if path.is_dir() and not path.is_symlink():
        for entry in path.iterdir():
            if entry.name not in CODEX_HOME_ALLOWLIST:
                unknown.append(entry.name)
            try:
                mode = stat.S_IMODE(entry.lstat().st_mode)
            except OSError:
                unsafe_modes.append(entry.name)
                continue
            if entry.is_dir() or entry.is_symlink() or mode != 0o600:
                unsafe_modes.append(entry.name)
    try:
        root_mode = stat.S_IMODE(path.lstat().st_mode)
        owner_ok = path.lstat().st_uid == os.getuid()
    except OSError:
        root_mode, owner_ok = 0, False
    ok = bool(
        path.is_dir()
        and not path.is_symlink()
        and root_mode == 0o700
        and owner_ok
        and under_owner_local
        and not cloud
        and not under_repo
        and not unknown
        and not unsafe_modes
    )
    return {
        "ok": ok,
        "path": str(resolved),
        "owner_only_0700": root_mode == 0o700 and owner_ok,
        "under_owner_local": under_owner_local,
        "outside_cloud": not cloud,
        "outside_repo": not under_repo,
        "unknown_files": sorted(unknown),
        "unsafe_files": sorted(unsafe_modes),
        "persistent_session_or_history": any(
            name.casefold() in {"sessions", "history.jsonl", "session.json"}
            for name in unknown
        ),
    }


def _owner_env(path: Path) -> Mapping[str, str]:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise RuntimeError("probe env must be owner-only 0600")
    from dotenv import dotenv_values

    return {
        str(key): str(value)
        for key, value in dotenv_values(path).items()
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
        # Sequential by construction: GigaAM starts only after Whisper exits.
        gigaam_code = (
            "import gigaam,sys;"
            "m=gigaam.load_model('v2_rnnt',device='cpu',fp16_encoder=False);"
            "m.transcribe(sys.argv[1])"
        )
        gigaam = subprocess.run(
            [str(config.python_executable), "-c", gigaam_code, str(sample)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=600,
        )
    return {
        "offline_whisper_synthetic_ok": mlx.returncode == 0,
        "offline_gigaam_synthetic_ok": gigaam.returncode == 0,
    }


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
    codex = inspect_codex_home(codex_home)
    google = {"google_spreadsheet_acl_ok": False, "google_metadata_readback_ok": False}
    try:
        google = dict(probe_google_readonly(config))
    except Exception:
        pass
    offline = {
        "offline_whisper_synthetic_ok": False,
        "offline_gigaam_synthetic_ok": False,
    }
    if run_offline_model_probes and parsed_config is not None:
        try:
            offline = dict(probe_offline_models(parsed_config))
        except Exception:
            pass
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
    }
    host_id = ""
    if parsed_config is not None:
        try:
            host_id = parsed_config.host_id_file.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    measurements_ok = _measurement_evidence_ok(
        evidence, expected_sha=expected_sha, host_id=host_id
    )
    configured_timezone = machine_timezone()
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
        "runtime_fingerprint_approved": runtime_observation.get("ok") is True and not fingerprint_errors,
        "isolated_codex_home_ok": codex["ok"] is True,
        "capacity_2x_and_memory_ok": capacity.get("capacity_ok") is True,
        "measurement_evidence_bound_and_fresh": measurements_ok,
        "timezone_is_europe_moscow": configured_timezone == "Europe/Moscow",
        "ffmpeg_present": shutil.which("ffmpeg") is not None,
        "ffprobe_present": shutil.which("ffprobe") is not None,
        **access,
    }
    return {
        "schema_version": "m1_mango_calls_access_probe_v1",
        "mode": (
            "read_only_access_plus_synthetic_offline_models"
            if run_offline_model_probes
            else "read_only_access_no_models_invoked"
        ),
        "host_readiness": "OK" if all(checks.values()) else "STOP",
        "checks": checks,
        "runtime_fingerprint_errors": fingerprint_errors,
        "runtime_observation_errors": runtime_observation.get("errors"),
        "codex_home": codex,
        "capacity": capacity,
        "machine": {
            "model": command_output(("sysctl", "-n", "hw.model")),
            "physical_memory_bytes": memory,
            "free_disk_bytes": disk.free,
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "timezone": configured_timezone,
        },
        "approved_runtime_fingerprint": approved_runtime_fingerprint(),
        "runs_asr": run_offline_model_probes,
        "runs_real_audio": False,
        "runs_resolve_analyze": False,
        "writes_external_systems": False,
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
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["host_readiness"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
