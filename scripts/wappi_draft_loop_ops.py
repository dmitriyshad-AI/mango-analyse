#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_amo_wappi_draft_loop as runner


TRACKED_ENV_PREFIXES = ("TELEGRAM_", "DRAFT_LOOP_")
SECRET_ENV_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|PASS|KEY|BEARER|ACCESS|AUTH)", re.IGNORECASE)
PROCESS_MARKER = "run_amo_wappi_draft_loop.py"
REQUIRED_RESOLVER_REASON_KEYS = (
    "amo_chat_event_sequence_unconfirmed",
    "amo_chat_event_rate_limited",
    "amo_chat_event_ambiguous",
    "brand_mismatch",
    "closed_lead",
    "max_phone_missing",
    "quarantined_pairs",
    "pending_notes",
    "quarantined",
    "pending",
    "not_enabled",
)
QUALITY_COLUMNS = (
    "created_at",
    "note_id",
    "lead_id",
    "contact_id",
    "profile_id",
    "chat_suffix",
    "message_id",
    "route",
    "safety_flags",
    "draft_text",
    "manager_reply_if_seen",
    "manual_label",
    "comment",
)


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    command: str


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    target = Path(path).expanduser()
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _row_datetime(row: Mapping[str, Any]) -> datetime | None:
    for key in ("created_at", "last_cycle_at", "draft_ts", "sent_ts"):
        parsed = _parse_iso_datetime(row.get(key))
        if parsed is not None:
            return parsed
    for key in ("timestamp", "employee_timestamp"):
        try:
            value = int(row.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return datetime.fromtimestamp(value, tz=timezone.utc)
    return None


def _read_json(path: Path | str) -> dict[str, Any]:
    target = Path(path).expanduser()
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _redact_env_value(key: str, value: Any) -> str:
    if SECRET_ENV_RE.search(str(key or "")):
        return "[REDACTED]"
    return str(value)


def filter_runtime_env(environ: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in sorted(environ.items()):
        if str(key).startswith(TRACKED_ENV_PREFIXES):
            result[str(key)] = _redact_env_value(str(key), value)
    return result


def _parse_ps_output(output: str) -> list[ProcessInfo]:
    result: list[ProcessInfo] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        result.append(ProcessInfo(pid=pid, ppid=ppid, command=parts[2]))
    return result


def list_processes() -> list[ProcessInfo]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return _parse_ps_output(completed.stdout)


def find_draft_loop_process(processes: Sequence[ProcessInfo] | None = None) -> ProcessInfo | None:
    candidates = []
    for process in processes if processes is not None else list_processes():
        command = process.command
        if PROCESS_MARKER in command and "wappi_draft_loop_ops.py" not in command:
            candidates.append(process)
    if not candidates:
        return None
    return sorted(candidates, key=_draft_loop_process_sort_key)[0]


def _draft_loop_process_sort_key(process: ProcessInfo) -> tuple[int, int]:
    command = process.command.casefold()
    wrapper_markers = ("screen ", "bash -lc", "sh -lc", "login -")
    is_wrapper = any(marker in command for marker in wrapper_markers)
    is_python_runner = "python" in command and PROCESS_MARKER.casefold() in command
    return (0 if is_python_runner and not is_wrapper else 1 if is_python_runner else 2, process.pid)


def _parse_null_environ(raw: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for chunk in raw.split(b"\0"):
        if not chunk or b"=" not in chunk:
            continue
        key, value = chunk.split(b"=", 1)
        result[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")
    return result


def read_process_environ(pid: int) -> tuple[dict[str, str], str]:
    proc_path = Path("/proc") / str(pid) / "environ"
    if proc_path.exists():
        try:
            return _parse_null_environ(proc_path.read_bytes()), "proc_environ"
        except OSError:
            pass
    completed = subprocess.run(
        ["ps", "eww", "-p", str(pid)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return {}, "unavailable"
    environ: dict[str, str] = {}
    for token in completed.stdout.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key.startswith(TRACKED_ENV_PREFIXES):
            environ[key] = value
    return environ, "ps_eww"


def _git_value(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _screen_context(process: ProcessInfo | None, processes: Sequence[ProcessInfo] | None = None) -> dict[str, Any]:
    if process is None:
        return {"detected": False}
    by_pid = {item.pid: item for item in (processes if processes is not None else list_processes())}
    chain: list[dict[str, Any]] = []
    current = process
    seen: set[int] = set()
    while current and current.pid not in seen:
        seen.add(current.pid)
        chain.append({"pid": current.pid, "ppid": current.ppid, "command": current.command})
        current = by_pid.get(current.ppid)
    screen_parent = next((item for item in chain if "screen" in str(item["command"]).casefold()), None)
    return {"detected": screen_parent is not None, "parent": screen_parent, "process_chain": chain[:6]}


def build_runtime_passport(
    *,
    repo_root: Path,
    config: runner.DraftLoopConfig,
    process_lister: Callable[[], Sequence[ProcessInfo]] | None = None,
    env_reader: Callable[[int], tuple[Mapping[str, Any], str]] | None = None,
) -> dict[str, Any]:
    processes = list(process_lister() if process_lister else list_processes())
    process = find_draft_loop_process(processes)
    runtime_env: Mapping[str, Any] = {}
    env_source = "process_not_found"
    if process is not None:
        runtime_env, env_source = (env_reader or read_process_environ)(process.pid)
    profile_channels = Counter(profile.channel for profile in config.profiles.values())
    return {
        "schema_version": "wappi_draft_loop_runtime_passport_v1_2026_06_25",
        "repo": {
            "path": str(repo_root),
            "branch": _git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git_value(repo_root, "rev-parse", "--short", "HEAD"),
        },
        "process": {
            "found": process is not None,
            "pid": process.pid if process else None,
            "command": process.command if process else "",
            "launch_path": _extract_launch_path(process.command if process else ""),
            "screen": _screen_context(process, processes),
        },
        "runtime_env": {
            "source": env_source,
            "values": filter_runtime_env(runtime_env),
        },
        "config": {
            "profiles_count": len(config.profiles),
            "profile_channels": dict(profile_channels),
            "pairs_count": len(config.pairs_snapshot()),
            "journal_path": str(config.journal_path),
            "heartbeat_path": str(config.heartbeat_path),
            "state_path": str(config.state_path),
            "manager_edit_log_path": str(config.manager_edit_log_path),
            "stop_path": str(config.stop_path),
            "stop_active": config.stop_path.expanduser().exists(),
        },
    }


def _extract_launch_path(command: str) -> str:
    for token in str(command or "").split():
        if token.endswith(PROCESS_MARKER):
            return token
    return ""


def _resolver_reason_from_row(row: Mapping[str, Any]) -> str:
    for key in ("auto_resolver_reason", "reason"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    candidate = row.get("auto_candidate")
    if isinstance(candidate, Mapping):
        for key in ("reason", "status"):
            value = str(candidate.get(key) or "").strip()
            if value:
                return value
    event = str(row.get("event") or "").strip()
    status = str(row.get("status") or "").strip()
    if event == "brand_pair_mismatch":
        return "brand_mismatch"
    if event == "pair_quarantined" or status == "quarantined":
        return "quarantined_pairs"
    if status == "note_pending":
        return "pending_notes"
    return ""


def _recent_rows(rows: Sequence[Mapping[str, Any]], *, now: datetime, window_hours: int) -> tuple[list[Mapping[str, Any]], int]:
    cutoff = now.astimezone(timezone.utc) - timedelta(hours=max(1, int(window_hours)))
    recent: list[Mapping[str, Any]] = []
    undated = 0
    for row in rows:
        row_dt = _row_datetime(row)
        if row_dt is None:
            undated += 1
            recent.append(row)
            continue
        if row_dt >= cutoff:
            recent.append(row)
    return recent, undated


def build_daily_report(
    *,
    journal_path: Path,
    heartbeat_path: Path,
    state_path: Path,
    now: datetime | None = None,
    window_hours: int = 24,
    heartbeat_fresh_sec: int = 600,
) -> dict[str, Any]:
    current_time = now or datetime.now(timezone.utc)
    rows, undated = _recent_rows(read_jsonl(journal_path), now=current_time, window_hours=window_hours)
    events = Counter(str(row.get("event") or "") for row in rows)
    statuses = Counter(str(row.get("status") or "") for row in rows)
    resolver_reasons = Counter({key: 0 for key in REQUIRED_RESOLVER_REASON_KEYS})
    for row in rows:
        reason = _resolver_reason_from_row(row)
        if reason:
            resolver_reasons[reason] += 1
    state = _read_json(state_path)
    quarantined = state.get("quarantined_pairs") if isinstance(state.get("quarantined_pairs"), Mapping) else {}
    pending = state.get("pending_notes") if isinstance(state.get("pending_notes"), Mapping) else {}
    resolver_reasons["quarantined_pairs"] += len(quarantined) + resolver_reasons["quarantined"]
    resolver_reasons["pending_notes"] += len(pending) + resolver_reasons["pending"]
    resolver_reasons["quarantined"] = resolver_reasons["quarantined_pairs"]
    resolver_reasons["pending"] = resolver_reasons["pending_notes"]
    heartbeat = _read_json(heartbeat_path)
    heartbeat_summary = heartbeat.get("summary") if isinstance(heartbeat.get("summary"), Mapping) else {}
    heartbeat_auto_counts = (
        heartbeat_summary.get("auto_resolver_counts") if isinstance(heartbeat_summary.get("auto_resolver_counts"), Mapping) else {}
    )
    for reason, count in heartbeat_auto_counts.items():
        try:
            resolver_reasons[str(reason)] += int(count or 0)
        except (TypeError, ValueError):
            continue
    last_cycle = _parse_iso_datetime(heartbeat.get("last_cycle_at"))
    age_sec = None if last_cycle is None else max(0.0, (current_time.astimezone(timezone.utc) - last_cycle).total_seconds())
    error_rows = [
        row
        for row in rows
        if row.get("error")
        or str(row.get("status") or "") in {"manual_review", "auth_error"}
        or "error" in str(row.get("event") or "").casefold()
    ]
    return {
        "schema_version": "wappi_draft_loop_daily_report_v1_2026_06_25",
        "window_hours": int(window_hours),
        "journal_path": str(journal_path),
        "heartbeat_path": str(heartbeat_path),
        "state_path": str(state_path),
        "alive": {
            "heartbeat_exists": bool(heartbeat),
            "last_cycle_at": heartbeat.get("last_cycle_at", ""),
            "age_sec": age_sec,
            "fresh": age_sec is not None and age_sec <= max(1, int(heartbeat_fresh_sec)),
            "status": heartbeat.get("status", ""),
        },
        "counts": {
            "rows_considered": len(rows),
            "undated_rows_considered": undated,
            "errors": len(error_rows),
            "draft_created": events.get("draft_created", 0),
            "notes_written": events.get("note_written", 0) + statuses.get("note_written", 0),
            "pair_missing": events.get("pair_missing", 0),
            "quarantined_pairs": len(quarantined),
            "pending_notes": len(pending),
        },
        "events": dict(events),
        "statuses": dict(statuses),
        "resolver_reasons": dict(resolver_reasons),
    }


def build_quality_rows(
    *,
    journal_rows: Sequence[Mapping[str, Any]],
    manager_edit_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, str]]:
    manager_by_key = {
        (
            str(row.get("profile_id") or ""),
            str(row.get("chat_id") or ""),
            str(row.get("message_id") or ""),
        ): row
        for row in manager_edit_rows
        if str(row.get("message_id") or "")
    }
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in journal_rows:
        if not str(row.get("bot_draft_text") or "").strip():
            continue
        if str(row.get("event") or "") not in {"draft_created", "note_written", "note_retried"}:
            continue
        key = (str(row.get("profile_id") or ""), str(row.get("chat_id") or ""), str(row.get("message_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        manager = manager_by_key.get(key, {})
        flags = row.get("safety_flags") or ()
        if isinstance(flags, str):
            flags_text = flags
        else:
            flags_text = "|".join(str(item) for item in flags)
        chat_id = str(row.get("chat_id") or "")
        result.append(
            {
                "created_at": str(row.get("created_at") or row.get("draft_ts") or ""),
                "note_id": str(row.get("note_id") or row.get("amo_note_id") or ""),
                "lead_id": str(row.get("lead_id") or ""),
                "contact_id": str(row.get("contact_id") or ""),
                "profile_id": str(row.get("profile_id") or ""),
                "chat_suffix": chat_id[-6:] if chat_id else "",
                "message_id": str(row.get("message_id") or ""),
                "route": str(row.get("route") or row.get("bot_route") or ""),
                "safety_flags": flags_text,
                "draft_text": str(row.get("bot_draft_text") or ""),
                "manager_reply_if_seen": str(manager.get("manager_sent_text") or ""),
                "manual_label": "",
                "comment": "",
            }
        )
    return result


def write_quality_csv(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(QUALITY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: str(row.get(column) or "") for column in QUALITY_COLUMNS})


def _common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profiles-file", type=Path, default=runner.DEFAULT_PROFILES_PATH)
    parser.add_argument("--pairs-file", type=Path, default=runner.DEFAULT_PAIRS_PATH)
    parser.add_argument("--auto-pairs-file", type=Path, default=runner.DEFAULT_AUTO_PAIRS_PATH)
    parser.add_argument("--phase1-config", type=Path, default=runner.DEFAULT_AMO_WAPPI_CONFIG_PATH)
    parser.add_argument("--local-dir", type=Path, default=runner.DEFAULT_DRAFT_LOOP_DIR)
    parser.add_argument("--stop-file", type=Path, default=runner.DEFAULT_STOP_PATH)
    parser.add_argument("--manager-outgoing-visible", choices=("unknown", "yes", "no"), default="unknown")
    parser.add_argument("--chat-limit", type=int, default=50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Wappi draft-loop ops tools.")
    sub = parser.add_subparsers(dest="command", required=True)
    passport = sub.add_parser("passport", help="Print runtime passport JSON.")
    _common_config_args(passport)
    passport.add_argument("--out", type=Path)

    daily = sub.add_parser("daily-report", help="Print daily journal/heartbeat report JSON.")
    _common_config_args(daily)
    daily.add_argument("--journal", type=Path)
    daily.add_argument("--heartbeat", type=Path)
    daily.add_argument("--state", type=Path)
    daily.add_argument("--window-hours", type=int, default=24)
    daily.add_argument("--heartbeat-fresh-sec", type=int, default=600)
    daily.add_argument("--out", type=Path)

    quality = sub.add_parser("quality-table", help="Build manual quality CSV from journal and manager edit log.")
    _common_config_args(quality)
    quality.add_argument("--journal", type=Path)
    quality.add_argument("--manager-edit-log", type=Path)
    quality.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> runner.DraftLoopConfig:
    return runner.build_config(args)


def _emit_json(payload: Mapping[str, Any], out: Path | None = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n"
    if out is None:
        print(text, end="")
    else:
        out.expanduser().parent.mkdir(parents=True, exist_ok=True)
        out.expanduser().write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = _config_from_args(args)
    if args.command == "passport":
        payload = build_runtime_passport(repo_root=Path.cwd(), config=config)
        _emit_json(payload, args.out)
        return 0
    if args.command == "daily-report":
        payload = build_daily_report(
            journal_path=(args.journal or config.journal_path).expanduser(),
            heartbeat_path=(args.heartbeat or config.heartbeat_path).expanduser(),
            state_path=(args.state or config.state_path).expanduser(),
            window_hours=args.window_hours,
            heartbeat_fresh_sec=args.heartbeat_fresh_sec,
        )
        _emit_json(payload, args.out)
        return 0
    if args.command == "quality-table":
        rows = build_quality_rows(
            journal_rows=read_jsonl((args.journal or config.journal_path).expanduser()),
            manager_edit_rows=read_jsonl((args.manager_edit_log or config.manager_edit_log_path).expanduser()),
        )
        write_quality_csv(rows, args.out.expanduser())
        print(json.dumps({"out": str(args.out.expanduser()), "rows": len(rows)}, ensure_ascii=False, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
