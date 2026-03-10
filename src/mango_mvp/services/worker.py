from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from mango_mvp.config import Settings
from mango_mvp.db import build_session_factory
from mango_mvp.services.analyze import AnalyzeService
from mango_mvp.services.resolve import ResolveService
from mango_mvp.services.sync_amocrm import AmoCRMSyncService
from mango_mvp.services.transcribe import TranscribeService


def run_worker(
    settings: Settings,
    *,
    stage_limit: int,
    once: bool,
    poll_sec: int | None = None,
    max_idle_cycles: int | None = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    session_factory = build_session_factory(settings)
    poll_interval = max(1, poll_sec if poll_sec is not None else settings.worker_poll_sec)
    max_idle = (
        max_idle_cycles
        if max_idle_cycles is not None
        else max(0, settings.worker_max_idle_cycles)
    )

    cycles = 0
    idle_cycles = 0
    totals = {
        "transcribe": {"processed": 0, "success": 0, "failed": 0},
        "resolve": {"processed": 0, "success": 0, "failed": 0},
        "analyze": {"processed": 0, "success": 0, "failed": 0},
        "sync": {"processed": 0, "success": 0, "failed": 0},
    }
    while True:
        cycles += 1
        with session_factory() as session:
            transcribe_result = TranscribeService(settings).run(session, limit=stage_limit)
        with session_factory() as session:
            resolve_result = ResolveService(settings).run(session, limit=stage_limit)
        with session_factory() as session:
            analyze_result = AnalyzeService(settings).run(session, limit=stage_limit)
        with session_factory() as session:
            sync_result = AmoCRMSyncService(settings).run(session, limit=stage_limit)

        cycle_payload = {
            "cycle": cycles,
            "transcribe": transcribe_result,
            "resolve": resolve_result,
            "analyze": analyze_result,
            "sync": sync_result,
        }
        if progress_callback is not None:
            progress_callback(cycle_payload)
        for stage, stage_result in (
            ("transcribe", transcribe_result),
            ("resolve", resolve_result),
            ("analyze", analyze_result),
            ("sync", sync_result),
        ):
            totals[stage]["processed"] += int(stage_result.get("processed", 0))
            totals[stage]["success"] += int(stage_result.get("success", 0))
            totals[stage]["failed"] += int(stage_result.get("failed", 0))

        cycle_work = (
            int(transcribe_result.get("processed", 0))
            + int(resolve_result.get("processed", 0))
            + int(analyze_result.get("processed", 0))
            + int(sync_result.get("processed", 0))
        )
        if once:
            return {
                "ok": True,
                "mode": "once",
                "cycles": cycles,
                "idle_cycles": idle_cycles,
                "totals": totals,
                "last_cycle": cycle_payload,
                "stop_reason": "once",
            }

        if cycle_work == 0:
            idle_cycles += 1
            if max_idle > 0 and idle_cycles >= max_idle:
                return {
                    "ok": True,
                    "mode": "loop",
                    "cycles": cycles,
                    "idle_cycles": idle_cycles,
                    "totals": totals,
                    "last_cycle": cycle_payload,
                    "stop_reason": "max_idle_cycles_reached",
                }
            time.sleep(poll_interval)
        else:
            idle_cycles = 0
