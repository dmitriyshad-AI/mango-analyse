#!/usr/bin/env python3
"""Deterministic M1 task watcher.

The watcher intentionally accepts only one fixed job shape: deploy a checked
``mango_clean_<sha>`` bundle and run ``run_telegram_dynamic_client_sim.py``.
It does not use LLMs and does not accept arbitrary scripts from task files.

Deprecated for new quick evals since TZ-155. Keep this watcher dormant; the
current manual path is git fetch/checkout plus ``build_job_manifest.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.channels.pilot_profile_runtime import (
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    ensure_canonical_pilot_profile,
)


WATCHER_VERSION = "m1_watcher_v1_2_2026_06_07"

DEFAULT_TESTS_ROOT = Path.home() / "Yandex.Disk.localized" / "OpenClaw" / "Actual Mango Tests"
DEFAULT_WORK_ROOT = Path.home() / "mango_m1_work"
DEFAULT_STATE_DIR = Path.home() / "m1_watcher"
DEFAULT_BUNDLE_WAIT_SECONDS = 2 * 60 * 60
DEFAULT_POLL_SECONDS = 180
DEFAULT_HEARTBEAT_SECONDS = 5 * 60
DEFAULT_STALL_SECONDS = 30 * 60
DEFAULT_MIN_FREE_BYTES = 5 * 1024 * 1024 * 1024
DEFAULT_READINESS_TIMEOUT_SECONDS = 20

TASK_NAME_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}_[A-Za-z0-9_.-]+\.task\.yaml$")
TASK_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{3,96}$")
BUNDLE_RE = re.compile(r"^mango_clean_[0-9a-f]{8}$")
ENV_NAME_RE = re.compile(r"^(TELEGRAM_[A-Z0-9_]+|DIALOGUE_CONTRACT_DEBUG_TRACE)$")
ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:/\-]{0,64}$")
CONFLICT_COPY_RE = re.compile(r"\s\(\d+\)\.task\.yaml$")
SET_SUFFIXES = (".jsonl", ".yaml")

# Deprecated since 2026-06-21: historical M1-only measurement stack.
# Task files provide only a delta over this mapping, so an empty env block still
# exercises the old full bot path unless the watcher is explicitly started with
# the V2 stack.
PRODUCTION_ENV_STACK: dict[str, str] = {
    "TELEGRAM_HANDOFF_TRACE": "1",
    "TELEGRAM_SEMANTIC_DIAGNOSIS_GUARD": "1",
    "TELEGRAM_PH2_TONE": "1",
    "TELEGRAM_PH2_OBJECTION": "1",
    "TELEGRAM_PH2_ANXIETY": "1",
    "TELEGRAM_A_TRAVEL_COMPOSE": "1",
    "TELEGRAM_OUTPUT_SANITIZER": "1",
    "TELEGRAM_A_ESTIMATE_MODE": "1",
    "TELEGRAM_A_FREE_NUMBER_GATE": "1",
    "TELEGRAM_Q_PARTIAL_YIELD": "1",
    "TELEGRAM_Q_CLARIFY_SCOPE": "1",
    "TELEGRAM_Q_USEFUL_HANDOFF": "1",
    "TELEGRAM_DIALOGUE_CONTRACT_PIPELINE": "1",
    "TELEGRAM_RULES_ENGINE_PLANNER_INTENT": "1",
    "TELEGRAM_STEP4_KEEP_ANSWER": "1",
    "TELEGRAM_STEP4_NUMBER_GROUNDING": "1",
    "TELEGRAM_SEMANTIC_OUTPUT_VERIFIER": "1",
    "TELEGRAM_TONE_WARM_FRAME": "1",
    "TELEGRAM_TONE_CLOSE_DETECT": "1",
    "TELEGRAM_TONE_SELL_PROMPT": "1",
    "TELEGRAM_TONE_RICH_FORMAT": "1",
    "TELEGRAM_COMPOSITE_CONTRACT_FIX": "1",
    "DIALOGUE_CONTRACT_DEBUG_TRACE": "1",
}

PRODUCTION_ENV_STACK_DEPRECATED_SINCE = "2026-06-21"
PRODUCTION_ENV_STACK_VERSION_LEGACY = "legacy"
PRODUCTION_ENV_STACK_VERSION_V2 = "v2"
DEFAULT_PRODUCTION_ENV_STACK_VERSION = PRODUCTION_ENV_STACK_VERSION_LEGACY

PRODUCTION_ENV_STACK_V2: dict[str, str] = {
    DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
}

_PRODUCTION_ENV_STACK_ALIASES = {
    "old": PRODUCTION_ENV_STACK_VERSION_LEGACY,
    "v1": PRODUCTION_ENV_STACK_VERSION_LEGACY,
    "legacy": PRODUCTION_ENV_STACK_VERSION_LEGACY,
    "v2": PRODUCTION_ENV_STACK_VERSION_V2,
}


class WatcherError(Exception):
    def __init__(self, code: str, detail: str = "", payload: Mapping[str, object] | None = None) -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail
        self.payload = dict(payload or {})


@dataclass(frozen=True)
class TaskSpec:
    id: str
    requires_bundle: str
    set: str
    set_sha256: str
    brain: str
    parallel: int
    max_hours: float
    judge_prompt_version: str
    env: dict[str, str]


@dataclass(frozen=True)
class RunOutcome:
    returncode: int
    started: bool = True
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""
    command: tuple[str, ...] = ()


Runner = Callable[[TaskSpec, Path, Path, tuple[str, ...], Mapping[str, str]], RunOutcome]
CommandProbe = Callable[[Sequence[str], Optional[Path], Optional[Mapping[str, str]], float], tuple[int, str, str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _tail_text(text: str, limit: int = 500) -> str:
    text = str(text or "").strip()
    return text[-limit:] if len(text) > limit else text


def run_command_probe(args: Sequence[str], cwd: Path | None, env: Mapping[str, str] | None, timeout: float) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return 124, stdout, stderr or f"timeout after {timeout}s"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def parse_task_yaml(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    env: dict[str, str] | None = None
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith((" ", "\t")):
            if env is None:
                raise WatcherError("task_schema_mismatch", f"unexpected indentation at line {lineno}")
            key, sep, value = raw_line.strip().partition(":")
            if not sep:
                raise WatcherError("task_schema_mismatch", f"bad env entry at line {lineno}")
            env[key.strip()] = _strip_quotes(value)
            continue
        key, sep, value = raw_line.partition(":")
        if not sep:
            raise WatcherError("task_schema_mismatch", f"bad key at line {lineno}")
        key = key.strip()
        value = value.strip()
        if key == "env":
            env = {}
            data["env"] = env
            if value:
                raise WatcherError("task_schema_mismatch", "env must be a mapping")
            continue
        env = None
        data[key] = _strip_quotes(value)
    return data


def validate_task_dict(data: Mapping[str, object]) -> TaskSpec:
    forbidden = {"script", "output"}
    if forbidden & set(data):
        raise WatcherError("task_schema_mismatch", "script/output fields are forbidden")
    required = {"id", "requires_bundle", "set", "set_sha256", "brain", "parallel", "max_hours"}
    missing = sorted(required - set(data))
    if missing:
        raise WatcherError("task_schema_mismatch", f"missing fields: {', '.join(missing)}")
    extra = sorted(set(data) - required - {"env", "judge_prompt_version"})
    if extra:
        raise WatcherError("task_schema_mismatch", f"unknown fields: {', '.join(extra)}")

    task_id = str(data["id"])
    if not TASK_ID_RE.fullmatch(task_id):
        raise WatcherError("task_schema_mismatch", "bad id")

    bundle = str(data["requires_bundle"])
    if not BUNDLE_RE.fullmatch(bundle):
        raise WatcherError("task_schema_mismatch", "bad requires_bundle")

    brain = str(data["brain"])
    if brain not in {"codex", "claude"}:
        raise WatcherError("task_schema_mismatch", "brain must be codex|claude")

    try:
        parallel = int(str(data["parallel"]))
        max_hours = float(str(data["max_hours"]))
    except ValueError as exc:
        raise WatcherError("task_schema_mismatch", "parallel/max_hours must be numeric") from exc
    if parallel < 1 or parallel > 4:
        raise WatcherError("task_schema_mismatch", "parallel must be 1..4")
    if max_hours <= 0 or max_hours > 9:
        raise WatcherError("task_schema_mismatch", "max_hours must be >0 and <=9")

    judge_prompt_version = str(data.get("judge_prompt_version") or "v9.1").strip().lower()
    if judge_prompt_version not in {"v2", "v9", "v9.1"}:
        raise WatcherError("task_schema_mismatch", "judge_prompt_version must be v2|v9|v9.1")

    env_raw = data.get("env") or {}
    if not isinstance(env_raw, Mapping):
        raise WatcherError("task_schema_mismatch", "env must be a mapping")
    env: dict[str, str] = {}
    for key, value in env_raw.items():
        key_s = str(key)
        value_s = str(value)
        if not ENV_NAME_RE.fullmatch(key_s):
            raise WatcherError("task_schema_mismatch", f"env key not allowed: {key_s}")
        if not ENV_VALUE_RE.fullmatch(value_s):
            raise WatcherError("task_schema_mismatch", f"env value not allowed: {key_s}")
        env[key_s] = value_s

    return TaskSpec(
        id=task_id,
        requires_bundle=bundle,
        set=str(data["set"]),
        set_sha256=str(data["set_sha256"]).strip(),
        brain=brain,
        parallel=parallel,
        max_hours=max_hours,
        judge_prompt_version=judge_prompt_version,
        env=env,
    )


def normalize_production_env_stack_version(value: str | None) -> str:
    key = str(value or DEFAULT_PRODUCTION_ENV_STACK_VERSION).strip().lower()
    version = _PRODUCTION_ENV_STACK_ALIASES.get(key)
    if not version:
        raise WatcherError("bad_production_env_stack", f"unknown production env stack: {value}")
    return version


def production_env_stack(version: str | None = None) -> dict[str, str]:
    normalized = normalize_production_env_stack_version(version)
    if normalized == PRODUCTION_ENV_STACK_VERSION_V2:
        return dict(PRODUCTION_ENV_STACK_V2)
    return dict(PRODUCTION_ENV_STACK)


def effective_task_env(delta: Mapping[str, str] | None = None, *, stack_version: str | None = None) -> dict[str, str]:
    env = production_env_stack(stack_version)
    if delta:
        env.update(delta)
    ensure_canonical_pilot_profile(environ=env)
    return env


def validate_set_path(tests_root: Path, rel_path: str, expected_sha256: str) -> Path:
    candidate = Path(rel_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise WatcherError("task_schema_mismatch", "set path must be relative and stay inside tests root")
    if candidate.suffix not in SET_SUFFIXES:
        raise WatcherError("task_schema_mismatch", "set must be .jsonl or .yaml")
    full_path = tests_root / candidate
    if full_path.is_symlink():
        raise WatcherError("task_schema_mismatch", "set symlink is forbidden")
    try:
        resolved = full_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise WatcherError("task_schema_mismatch", "set file missing") from exc
    root_resolved = tests_root.resolve()
    if root_resolved not in (resolved, *resolved.parents):
        raise WatcherError("task_schema_mismatch", "set path escapes tests root")
    if sha256_file(resolved) != expected_sha256:
        raise WatcherError("task_schema_mismatch", "set sha256 mismatch")
    return resolved


def load_json(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def verify_bundle_manifest(bundle_dir: Path) -> dict[str, object]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise WatcherError("bundle_waiting", "manifest.json is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WatcherError("bundle_waiting", "manifest.json is not valid JSON") from exc
    files = manifest.get("files")
    if not isinstance(files, list):
        raise WatcherError("bundle_waiting", "manifest files must be a list")

    listed: dict[str, tuple[int, str]] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise WatcherError("bundle_waiting", "manifest file entry must be an object")
        rel = str(item.get("path", ""))
        if not rel or rel.startswith("/") or ".." in Path(rel).parts or rel == "manifest.json":
            raise WatcherError("bundle_waiting", f"bad manifest path: {rel}")
        listed[rel] = (int(item.get("size", -1)), str(item.get("sha256", "")))

    actual: dict[str, Path] = {}
    for path in bundle_dir.rglob("*"):
        if not path.is_file() or path.name == "manifest.json":
            continue
        rel = path.relative_to(bundle_dir).as_posix()
        actual[rel] = path
    if int(manifest.get("file_count", -1)) != len(listed) or set(actual) != set(listed):
        raise WatcherError("bundle_waiting", "manifest file set mismatch")
    for rel, path in actual.items():
        size, expected_hash = listed[rel]
        if path.stat().st_size != size:
            raise WatcherError("bundle_waiting", f"size mismatch: {rel}")
        if sha256_file(path) != expected_hash:
            raise WatcherError("bundle_waiting", f"sha256 mismatch: {rel}")
    return manifest


def parse_bundle_info(bundle_dir: Path) -> dict[str, str]:
    info_path = bundle_dir / "BUNDLE_INFO.txt"
    if not info_path.exists():
        raise WatcherError("bundle_info_missing", "BUNDLE_INFO.txt is missing")
    info: dict[str, str] = {}
    for line in info_path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    if not info.get("kb_snapshot"):
        raise WatcherError("bundle_info_missing", "kb_snapshot missing in BUNDLE_INFO.txt")
    return info


class M1Watcher:
    def __init__(
        self,
        tests_root: Path = DEFAULT_TESTS_ROOT,
        work_root: Path = DEFAULT_WORK_ROOT,
        state_dir: Path = DEFAULT_STATE_DIR,
        *,
        bundle_wait_seconds: int = DEFAULT_BUNDLE_WAIT_SECONDS,
        heartbeat_seconds: int = DEFAULT_HEARTBEAT_SECONDS,
        stall_seconds: int = DEFAULT_STALL_SECONDS,
        min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
        readiness_timeout_seconds: int = DEFAULT_READINESS_TIMEOUT_SECONDS,
        runner: Runner | None = None,
        command_probe: CommandProbe | None = None,
        process_alive: Callable[[int], bool] | None = None,
        production_stack_version: str = DEFAULT_PRODUCTION_ENV_STACK_VERSION,
    ) -> None:
        self.tests_root = tests_root
        self.tasks_root = tests_root / "tasks"
        self.work_root = work_root
        self.state_dir = state_dir
        self.state_path = state_dir / "state.json"
        self.bundle_wait_seconds = bundle_wait_seconds
        self.heartbeat_seconds = max(1, int(heartbeat_seconds))
        self.stall_seconds = max(1, int(stall_seconds))
        self.min_free_bytes = max(0, int(min_free_bytes))
        self.readiness_timeout_seconds = max(1, int(readiness_timeout_seconds))
        self.runner = runner
        self.command_probe = command_probe or run_command_probe
        self.process_alive = process_alive or self._process_alive
        self.production_stack_version = normalize_production_env_stack_version(production_stack_version)

    @property
    def inbox(self) -> Path:
        return self.tasks_root / "_inbox_m1"

    @property
    def running(self) -> Path:
        return self.tasks_root / "_running"

    @property
    def done(self) -> Path:
        return self.tasks_root / "_done"

    @property
    def failed(self) -> Path:
        return self.tasks_root / "_failed"

    @property
    def status_path(self) -> Path:
        return self.tasks_root / "watcher_status.txt"

    @property
    def logs_dir(self) -> Path:
        return self.state_dir / "logs"

    def ensure_layout(self) -> None:
        for path in (
            self.inbox,
            self.running,
            self.done,
            self.failed,
            self.tests_root / "runs",
            self.work_root,
            self.state_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> dict[str, object]:
        state = load_json(self.state_path, {})
        return state if isinstance(state, dict) else {}

    def save_state(self, state: Mapping[str, object]) -> None:
        write_json_atomic(self.state_path, state)

    def _today_log_path(self) -> Path:
        return self.logs_dir / f"{datetime.now(timezone.utc).astimezone().date().isoformat()}.log"

    def _log_event(self, event: str, **fields: object) -> None:
        path = self._today_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "time": utc_now(),
            "watcher_version": WATCHER_VERSION,
            "event": event,
            **fields,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _log_tail(self, lines: int = 5) -> list[str]:
        path = self._today_log_path()
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]

    def write_status(self, status: str, detail: str = "", current_task: str = "") -> None:
        self._log_event("status", status=status, detail=detail, current_task=current_task)
        log_tail = self._log_tail()
        status_lines = [
            f"status: {status}",
            f"detail: {detail}",
            f"current_task: {current_task}",
            f"watcher_version: {WATCHER_VERSION}",
            f"host: {socket.gethostname()}",
            f"updated_at: {utc_now()}",
            f"log_path: {self._today_log_path()}",
            "log_tail:",
        ]
        status_lines.extend(f"  {line}" for line in log_tail)
        status_lines.append("")
        write_text_atomic(
            self.status_path,
            "\n".join(status_lines),
        )

    def _heartbeat_path(self, task_id: str) -> Path:
        return self.running / f"{task_id}.heartbeat"

    def _heartbeat_counter(self, task_id: str) -> int:
        current = load_json(self._heartbeat_path(task_id), {})
        if isinstance(current, Mapping):
            try:
                return int(current.get("counter") or 0) + 1
            except (TypeError, ValueError):
                return 1
        return 1

    def _write_heartbeat(self, task_id: str, stage: str, out_dir: Path | None = None) -> None:
        transcript_path = out_dir / "dynamic_dialog_transcripts.jsonl" if out_dir else None
        transcript_exists = bool(transcript_path and transcript_path.exists())
        age_seconds: float | None = None
        stall = False
        if transcript_path and transcript_exists:
            age_seconds = max(0.0, time.time() - transcript_path.stat().st_mtime)
            stall = age_seconds >= self.stall_seconds
        payload = {
            "task_id": task_id,
            "counter": self._heartbeat_counter(task_id),
            "stage": stage,
            "updated_at": utc_now(),
            "transcript_path": str(transcript_path) if transcript_path else "",
            "transcript_exists": transcript_exists,
            "transcript_mtime_age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
            "stall": stall,
        }
        write_json_atomic(self._heartbeat_path(task_id), payload)
        self._log_event("heartbeat", task_id=task_id, stage=stage, counter=payload["counter"], stall=stall)

    def _cli_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        for name, args in {
            "codex": ("codex", "--version"),
            "claude": ("claude", "--version"),
        }.items():
            code, stdout, stderr = self.command_probe(args, None, None, self.readiness_timeout_seconds)
            text = _tail_text(stdout or stderr, 200)
            versions[name] = text if code == 0 and text else f"unavailable(returncode={code}): {_tail_text(stderr or stdout, 160)}"
        return versions

    def _base_run_env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        env = os.environ.copy()
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"})
        env.update(effective_task_env(extra, stack_version=self.production_stack_version))
        return env

    def _readiness_checks(self, spec: TaskSpec, deploy_dir: Path, env: Mapping[str, str]) -> dict[str, object]:
        readiness: dict[str, object] = {}

        code, stdout, stderr = self.command_probe(
            (sys.executable, "-c", "import mango_mvp"),
            deploy_dir,
            env,
            self.readiness_timeout_seconds,
        )
        readiness["import_mango_mvp"] = {"ok": code == 0, "returncode": code, "stderr_tail": _tail_text(stderr, 300)}
        if code != 0:
            raise WatcherError("readiness_failed", f"import mango_mvp failed: {_tail_text(stderr or stdout, 200)}", payload=readiness)

        usage = shutil.disk_usage(self.work_root)
        readiness["disk_free_bytes"] = usage.free
        readiness["disk_min_free_bytes"] = self.min_free_bytes
        readiness["disk_ok"] = usage.free >= self.min_free_bytes
        if usage.free < self.min_free_bytes:
            raise WatcherError("readiness_failed", f"free disk below threshold: {usage.free} < {self.min_free_bytes}", payload=readiness)

        if spec.brain == "claude":
            code, stdout, stderr = self.command_probe(
                ("claude", "auth", "status", "--json"),
                deploy_dir,
                None,
                self.readiness_timeout_seconds,
            )
            readiness["claude_auth_status"] = {
                "ok": code == 0,
                "returncode": code,
                "stdout_tail": _tail_text(stdout, 300),
                "stderr_tail": _tail_text(stderr, 300),
            }
            if code != 0:
                raise WatcherError("readiness_failed", f"claude auth status failed: {_tail_text(stderr or stdout, 200)}", payload=readiness)
        return readiness

    def unacked_executed_count(self) -> int:
        ack_files = sorted(self.tasks_root.glob("ACK_*.md"), key=lambda p: p.stat().st_mtime)
        ack_mtime = ack_files[-1].stat().st_mtime if ack_files else 0.0
        count = 0
        for folder in (self.done, self.failed):
            for report in folder.glob("*.report.md"):
                if report.stat().st_mtime <= ack_mtime:
                    continue
                text = report.read_text(encoding="utf-8", errors="replace")
                if re.search(r"^executed:\s*true\s*$", text, re.MULTILINE):
                    count += 1
        return count

    def terminal_report_exists(self, task_id: str) -> bool:
        return (self.done / f"{task_id}.report.md").exists() or (self.failed / f"{task_id}.report.md").exists()

    def running_tasks(self) -> list[Path]:
        return sorted(self.running.glob("*.task.yaml"))

    def _process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _handle_existing_running(self) -> bool:
        tasks = self.running_tasks()
        if not tasks:
            return False
        task_path = tasks[0]
        task_id = task_path.name.removesuffix(".task.yaml")
        claimed = load_json(self.running / f"{task_id}.claimed", {})
        pgid = int(claimed.get("pgid") or claimed.get("pid") or 0) if isinstance(claimed, Mapping) else 0
        if pgid and self.process_alive(pgid):
            self.write_status("orphan_run_alive", current_task=task_id)
            return True
        self._finish_running_task(task_id, "interrupted", "watcher restarted and run process is dead", executed=False)
        return True

    def _task_signature_stable(self, task_path: Path, state: dict[str, object]) -> bool:
        sigs = state.setdefault("file_signatures", {})
        assert isinstance(sigs, dict)
        stat = task_path.stat()
        current = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        record = sigs.get(task_path.name)
        if isinstance(record, Mapping) and record.get("size") == current["size"] and record.get("mtime_ns") == current["mtime_ns"]:
            sigs[task_path.name] = {**current, "stable_count": int(record.get("stable_count", 0)) + 1}
            return int(record.get("stable_count", 0)) + 1 >= 1
        sigs[task_path.name] = {**current, "stable_count": 0}
        return False

    def _ready_marker_ok(self, task_path: Path) -> bool:
        ready_path = task_path.with_suffix(task_path.suffix + ".ready")
        if not ready_path.exists():
            return False
        expected = ready_path.read_text(encoding="utf-8", errors="replace").strip()
        expected = expected.removeprefix("sha256:").strip()
        if expected != sha256_file(task_path):
            raise WatcherError("ready_sha_mismatch", "ready marker sha256 does not match yaml")
        return True

    def _parse_wait_or_fail(self, task_path: Path, exc: WatcherError) -> bool:
        state = self.load_state()
        waits = state.setdefault("parse_error_waits", {})
        assert isinstance(waits, dict)
        sha = sha256_file(task_path)
        record = waits.get(task_path.name)
        if isinstance(record, Mapping) and record.get("sha256") == sha:
            count = int(record.get("count", 0)) + 1
        else:
            count = 1
        waits[task_path.name] = {"sha256": sha, "count": count, "detail": exc.detail}
        self.save_state(state)
        if count <= 3:
            self.write_status("task_parse_waiting", f"{task_path.name}: {exc.detail} ({count}/3)")
            return True
        waits.pop(task_path.name, None)
        self.save_state(state)
        return False

    def _parse_task_spec_or_wait(self, task_path: Path) -> TaskSpec | None:
        try:
            data = parse_task_yaml(task_path)
        except WatcherError as exc:
            if self._parse_wait_or_fail(task_path, exc):
                return None
            self._fail_inbox_task(task_path, exc.code, exc.detail)
            return None
        try:
            return validate_task_dict(data)
        except WatcherError as exc:
            self._fail_inbox_task(task_path, exc.code, exc.detail)
            return None

    def _next_task_path(self) -> Path | None:
        state = self.load_state()
        chosen: Path | None = None
        for task_path in sorted(self.inbox.glob("*.task.yaml")):
            if CONFLICT_COPY_RE.search(task_path.name):
                self.write_status("idle", f"ignored conflict copy: {task_path.name}")
                continue
            if not TASK_NAME_RE.fullmatch(task_path.name):
                self._fail_inbox_task(task_path, "task_schema_mismatch", "bad task filename")
                continue
            ready_path = task_path.with_suffix(task_path.suffix + ".ready")
            if not ready_path.exists():
                continue
            if not self._task_signature_stable(task_path, state):
                continue
            try:
                self._ready_marker_ok(task_path)
            except WatcherError as exc:
                self._fail_inbox_task(task_path, exc.code, exc.detail)
                continue
            chosen = task_path
            break
        self.save_state(state)
        return chosen

    def _claim(self, task_path: Path, spec: TaskSpec) -> tuple[Path, Path]:
        ready_path = task_path.with_suffix(task_path.suffix + ".ready")
        running_task = self.running / f"{spec.id}.task.yaml"
        running_ready = self.running / f"{spec.id}.task.yaml.ready"
        task_path.replace(running_task)
        ready_path.replace(running_ready)
        write_json_atomic(
            self.running / f"{spec.id}.claimed",
            {
                "task_id": spec.id,
                "watcher_pid": os.getpid(),
                "host": socket.gethostname(),
                "claimed_at": utc_now(),
                "watcher_version": WATCHER_VERSION,
            },
        )
        return running_task, running_ready

    def _fail_inbox_task(self, task_path: Path, status: str, detail: str) -> None:
        try:
            data = parse_task_yaml(task_path)
            task_id = str(data.get("id") or task_path.stem.removesuffix(".task"))
        except Exception:
            task_id = task_path.stem.removesuffix(".task")
        report_path = self.failed / f"{task_id}.report.md"
        self._write_report(report_path, task_id, status, detail, executed=False)
        failed_task = self.failed / task_path.name
        task_path.replace(failed_task)
        ready = task_path.with_suffix(task_path.suffix + ".ready")
        if ready.exists():
            ready.replace(self.failed / ready.name)

    def _finish_running_task(self, task_id: str, status: str, detail: str, *, executed: bool, summary: Mapping[str, object] | None = None) -> None:
        self._write_report(self.failed / f"{task_id}.report.md", task_id, status, detail, executed=executed, summary=summary)
        for path in self.running.glob(f"{task_id}*"):
            path.replace(self.failed / path.name)
        self.write_status(status, detail, current_task=task_id)

    def _write_report(
        self,
        path: Path,
        task_id: str,
        status: str,
        detail: str,
        *,
        executed: bool,
        summary: Mapping[str, object] | None = None,
        command: tuple[str, ...] = (),
        bundle_info: Mapping[str, str] | None = None,
        cli_versions: Mapping[str, str] | None = None,
        readiness: Mapping[str, object] | None = None,
        results_path: Path | None = None,
        task_env_delta: Mapping[str, str] | None = None,
        effective_env: Mapping[str, str] | None = None,
        production_stack_version: str | None = None,
    ) -> None:
        lines = [
            f"# M1 watcher report: {task_id}",
            "",
            f"status: {status}",
            f"detail: {detail}",
            f"executed: {'true' if executed else 'false'}",
            f"watcher_version: {WATCHER_VERSION}",
            f"host: {socket.gethostname()}",
            f"created_at: {utc_now()}",
        ]
        if bundle_info:
            lines.extend([f"bundle_head: {bundle_info.get('head', '')}", f"kb_snapshot: {bundle_info.get('kb_snapshot', '')}"])
        if cli_versions:
            lines.extend(
                [
                    f"codex_cli_version: {cli_versions.get('codex', '')}",
                    f"claude_cli_version: {cli_versions.get('claude', '')}",
                ]
            )
        if readiness:
            lines.append("readiness:")
            for key, value in readiness.items():
                lines.append(f"  {key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
        if command:
            lines.append("command: " + " ".join(command))
        if effective_env is not None:
            lines.append("env:")
            lines.append("  production_stack: true")
            lines.append(f"  production_stack_version: {production_stack_version or DEFAULT_PRODUCTION_ENV_STACK_VERSION}")
            lines.append(f"  task_delta: {json.dumps(dict(task_env_delta or {}), ensure_ascii=False, sort_keys=True)}")
            lines.append(f"  effective_task_env: {json.dumps(dict(effective_env), ensure_ascii=False, sort_keys=True)}")
        if results_path:
            lines.append(f"results_path: {results_path}")
        if summary:
            lines.append("summary:")
            for key in ("overall_verdict", "hard_gates_passed", "total_turns", "llm_calls"):
                if key in summary:
                    value = summary[key]
                    lines.append(f"  {key}: {json.dumps(value, ensure_ascii=False)}")
        lines.append("")
        write_text_atomic(path, "\n".join(lines))

    def _wait_or_fail_bundle(self, bundle_id: str, detail: str) -> bool:
        state = self.load_state()
        waits = state.setdefault("bundle_waits", {})
        assert isinstance(waits, dict)
        now = time.time()
        first_seen = float(waits.get(bundle_id, now))
        waits[bundle_id] = first_seen
        self.save_state(state)
        if now - first_seen < self.bundle_wait_seconds:
            self.write_status("bundle_waiting", detail)
            return True
        return False

    def _deploy_bundle(self, bundle_id: str) -> tuple[Path, dict[str, str]]:
        source = self.tests_root / bundle_id
        if not source.exists():
            raise WatcherError("bundle_waiting", "bundle directory is missing")
        try:
            verify_bundle_manifest(source)
        except WatcherError as exc:
            if exc.code == "bundle_waiting" and self._wait_or_fail_bundle(bundle_id, exc.detail):
                raise
            raise WatcherError("bundle_incomplete", exc.detail) from exc
        info = parse_bundle_info(source)
        target = self.work_root / bundle_id
        marker = target / ".deployed_ok"
        if marker.exists():
            return target, info
        if target.exists():
            raise WatcherError("deploy_incomplete", "bundle target exists without .deployed_ok")
        shutil.copytree(source, target, symlinks=False)
        write_text_atomic(marker, f"deployed_at: {utc_now()}\nsource: {source}\n")
        return target, info

    def _build_command(self, spec: TaskSpec, deploy_dir: Path, set_path: Path, out_dir: Path, snapshot_rel: str) -> tuple[str, ...]:
        command = [
            sys.executable,
            "scripts/run_telegram_dynamic_client_sim.py",
            "--scenarios",
            str(set_path),
            "--snapshot",
            snapshot_rel,
            "--bot-mode",
            spec.brain,
            "--memory-mode",
            "codex",
            "--memory-reasoning",
            "low",
            "--semantic-mode",
            "codex",
            "--semantic-reasoning",
            "medium",
            "--parallel",
            str(spec.parallel),
            "--judge-prompt-version",
            spec.judge_prompt_version,
            "--out-dir",
            str(out_dir),
        ]
        if spec.brain == "claude":
            command.extend(["--timeout-sec", "270"])
        return tuple(command)

    def _run_subprocess(self, spec: TaskSpec, deploy_dir: Path, out_dir: Path, command: tuple[str, ...], env: Mapping[str, str]) -> RunOutcome:
        self._write_heartbeat(spec.id, "running_start", out_dir)
        proc = subprocess.Popen(
            command,
            cwd=deploy_dir,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        claimed_path = self.running / f"{spec.id}.claimed"
        claimed = load_json(claimed_path, {})
        if not isinstance(claimed, dict):
            claimed = {}
        claimed.update({"run_pid": proc.pid, "pgid": os.getpgid(proc.pid), "run_started_at": utc_now(), "command": list(command)})
        write_json_atomic(claimed_path, claimed)
        deadline = time.monotonic() + spec.max_hours * 3600
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, spec.max_hours * 3600)
                try:
                    stdout, stderr = proc.communicate(timeout=min(float(self.heartbeat_seconds), remaining))
                    self._write_heartbeat(spec.id, "running_finished", out_dir)
                    return RunOutcome(proc.returncode, True, False, stdout, stderr, command)
                except subprocess.TimeoutExpired:
                    self._write_heartbeat(spec.id, "running", out_dir)
        except subprocess.TimeoutExpired:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                stdout, stderr = proc.communicate(timeout=15)
            self._write_heartbeat(spec.id, "timeout", out_dir)
            return RunOutcome(proc.returncode if proc.returncode is not None else -signal.SIGTERM, True, True, stdout, stderr, command)

    def _execute(self, task_path: Path, spec: TaskSpec) -> str:
        cli_versions = self._cli_versions()
        if self.terminal_report_exists(spec.id):
            self._fail_inbox_task(task_path, "duplicate_id", "task id already has terminal report")
            return "duplicate_id"
        try:
            validate_set_path(self.tests_root, spec.set, spec.set_sha256)
        except WatcherError as exc:
            self._fail_inbox_task(task_path, exc.code, exc.detail)
            return exc.code
        running_task, _ = self._claim(task_path, spec)
        self._write_heartbeat(spec.id, "claimed")
        self.write_status("running", current_task=spec.id)
        try:
            self._write_heartbeat(spec.id, "deploying")
            deploy_dir, bundle_info = self._deploy_bundle(spec.requires_bundle)
        except WatcherError as exc:
            terminal = exc.code == "bundle_incomplete"
            if terminal:
                self._write_report(
                    self.failed / f"{spec.id}.report.md",
                    spec.id,
                    exc.code,
                    exc.detail,
                    executed=False,
                    cli_versions=cli_versions,
                )
                for path in self.running.glob(f"{spec.id}*"):
                    path.replace(self.failed / path.name)
                self.write_status(exc.code, exc.detail, current_task=spec.id)
            return exc.code

        set_path = (self.tests_root / spec.set).resolve()
        work_out_dir = deploy_dir / "runs" / spec.id
        command = self._build_command(spec, deploy_dir, set_path, work_out_dir, bundle_info["kb_snapshot"])
        task_env = effective_task_env(spec.env, stack_version=self.production_stack_version)
        env = self._base_run_env(spec.env)

        try:
            self._write_heartbeat(spec.id, "readiness")
            readiness = self._readiness_checks(spec, deploy_dir, env)
        except WatcherError as exc:
            readiness = exc.payload
            self._write_heartbeat(spec.id, "readiness_failed", work_out_dir)
            self._write_report(
                self.failed / f"{spec.id}.report.md",
                spec.id,
                exc.code,
                exc.detail,
                executed=False,
                command=command,
                bundle_info=bundle_info,
                cli_versions=cli_versions,
                readiness=readiness,
                task_env_delta=spec.env,
                effective_env=task_env,
                production_stack_version=self.production_stack_version,
            )
            for path in self.running.glob(f"{spec.id}*"):
                path.replace(self.failed / path.name)
            self.write_status(exc.code, exc.detail, current_task=spec.id)
            return exc.code

        runner = self.runner or self._run_subprocess
        self._write_heartbeat(spec.id, "running", work_out_dir)
        self._log_event("run_start", task_id=spec.id, brain=spec.brain, command=" ".join(command))
        outcome = runner(spec, deploy_dir, work_out_dir, command, env)
        status = "timeout" if outcome.timed_out else ("success" if outcome.returncode == 0 else "run_failed")
        yandex_results = self.tests_root / "runs" / spec.id
        self._write_heartbeat(spec.id, "copy_results", work_out_dir)
        if work_out_dir.exists() and not yandex_results.exists():
            shutil.copytree(work_out_dir, yandex_results, symlinks=False)
        summary = load_json(work_out_dir / "dynamic_summary.json", {})
        if not isinstance(summary, Mapping):
            summary = {}
        config_validity = summary.get("config_validity") if isinstance(summary.get("config_validity"), Mapping) else {}
        if config_validity.get("invalid"):
            status = "config_invalid"

        terminal_dir = self.done if status == "success" else self.failed
        self._write_report(
            terminal_dir / f"{spec.id}.report.md",
            spec.id,
            status,
            "",
            executed=outcome.started,
            summary=summary,
            command=outcome.command or command,
            bundle_info=bundle_info,
            cli_versions=cli_versions,
            readiness=readiness,
            results_path=yandex_results,
            task_env_delta=spec.env,
            effective_env=task_env,
            production_stack_version=self.production_stack_version,
        )
        self._write_heartbeat(spec.id, status, work_out_dir)
        for path in self.running.glob(f"{spec.id}*"):
            path.replace(terminal_dir / path.name)
        self._log_event("run_finish", task_id=spec.id, status=status, returncode=outcome.returncode, timed_out=outcome.timed_out)
        self.write_status(status, current_task=spec.id)
        return status

    def process_once(self) -> str:
        self.ensure_layout()
        if (self.tasks_root / "STOP").exists():
            self.write_status("stopped")
            return "stopped"
        if self.unacked_executed_count() >= 3:
            self.write_status("awaiting_ack", "3 unconfirmed executed tasks")
            return "awaiting_ack"
        if self._handle_existing_running():
            return "running_present"
        task_path = self._next_task_path()
        if task_path is None:
            self.write_status("idle")
            return "idle"
        spec = self._parse_task_spec_or_wait(task_path)
        if spec is None:
            return "task_parse_waiting" if task_path.exists() else "task_schema_mismatch"
        return self._execute(task_path, spec)

    def loop(self, poll_seconds: int = DEFAULT_POLL_SECONDS) -> None:
        while True:
            self.process_once()
            time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic M1 task watcher.")
    parser.add_argument("--tests-root", type=Path, default=DEFAULT_TESTS_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument(
        "--production-env-stack",
        default=DEFAULT_PRODUCTION_ENV_STACK_VERSION,
        choices=sorted(_PRODUCTION_ENV_STACK_ALIASES),
        help="Measurement stack version. Default keeps the historical legacy stack; use v2 explicitly for clean direct-path profile.",
    )
    parser.add_argument("--once", action="store_true", help="Process one polling cycle and exit.")
    args = parser.parse_args(argv)
    watcher = M1Watcher(
        args.tests_root,
        args.work_root,
        args.state_dir,
        production_stack_version=args.production_env_stack,
    )
    if args.once:
        print(watcher.process_once())
        return 0
    watcher.loop(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
