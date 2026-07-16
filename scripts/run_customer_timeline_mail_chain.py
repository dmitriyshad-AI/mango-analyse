#!/usr/bin/env python3
"""Run mail download -> process -> import as one fail-fast chain."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_customer_timeline_codex_task as codex_task  # noqa: E402

STAGES = ("mail-download", "mail-process", "mail-import")


@dataclass(frozen=True)
class StageRun:
    task: str
    rc: int
    payload: Mapping[str, Any]


@contextmanager
def chain_lock(path: Path) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def run_stage_subprocess(task: str) -> StageRun:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_customer_timeline_codex_task.py"), "--task", task],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return StageRun(task=task, rc=completed.returncode, payload=codex_task.parse_last_json(completed.stdout or ""))


def stage_preflight_stop_reason(task: str) -> str:
    return "" if task == "mail-download" else codex_task.build_task_spec(task, tallanto_phone_limit=250).stop_reason


def chain_report(status: str, result: str, stop_reason: str, started_at: str, stages: list[dict[str, Any]]) -> Mapping[str, Any]:
    return {
        "schema_version": "customer_timeline_mail_chain_v1",
        "status": status,
        "result": result,
        "stop_reason": stop_reason,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "stages": stages,
    }


def _stage_status(run: StageRun) -> tuple[str, str]:
    status = str(run.payload.get("status") or "").strip()
    reason = str(run.payload.get("stop_reason") or run.payload.get("reason") or "").strip()
    if status == "ok" and run.rc == 0:
        return "ok", ""
    if status in {"already_running", "locked"}:
        return "stopped", "already_running"
    if not reason:
        reason = status or str(run.payload.get("error") or f"command_rc={run.rc}")
    return "stopped", reason


def run_chain(
    *,
    lock_path: Path,
    runner: Callable[[str], StageRun] = run_stage_subprocess,
    preflight: Callable[[str], str] = stage_preflight_stop_reason,
) -> Mapping[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    stages: list[dict[str, Any]] = []
    with chain_lock(lock_path) as acquired:
        if not acquired:
            return chain_report("stopped", "already_running", "already_running", started_at, stages)
        for task in STAGES:
            stop_reason = preflight(task)
            if stop_reason:
                stages.append({"task": task, "status": "stopped", "stop_reason": stop_reason, "started": False})
                return chain_report("stopped", "stopped", f"{task}:{stop_reason}", started_at, stages)
            run = runner(task)
            status, reason = _stage_status(run)
            stages.append(
                {
                    "task": task,
                    "status": status,
                    "stop_reason": reason,
                    "rc": run.rc,
                    "started": True,
                }
            )
            if status != "ok":
                return chain_report("stopped", "stopped", f"{task}:{reason}", started_at, stages)
    return chain_report("ok", "ok", "", started_at, stages)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-path", default=str(codex_task.MAIL_STATE_DIR / "mail_chain.lock"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_chain(lock_path=Path(args.lock_path).expanduser().resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
