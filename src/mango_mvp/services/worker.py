from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Callable, Dict, Optional

from mango_mvp.config import Settings
from mango_mvp.db import build_session_factory
from mango_mvp.services.analyze import AnalyzeService
from mango_mvp.services.controlled_call_scope import (
    controlled_worker_parent_lifeline,
    enforce_controlled_worker_stages,
)
from mango_mvp.services.resolve import ResolveService
from mango_mvp.services.sync_amocrm import AmoCRMSyncService
from mango_mvp.services.transcribe import TranscribeService

AVAILABLE_PIPELINE_STAGE_ORDER = (
    "transcribe",
    "backfill-second-asr",
    "resolve",
    "analyze",
    "sync",
)

DEFAULT_PIPELINE_STAGE_ORDER = tuple(
    stage for stage in AVAILABLE_PIPELINE_STAGE_ORDER if stage != "sync"
)


def normalize_pipeline_stages(stages: Optional[list[str]]) -> list[str]:
    if not stages:
        return list(DEFAULT_PIPELINE_STAGE_ORDER)

    aliases = {
        "backfill": "backfill-second-asr",
        "secondary-asr": "backfill-second-asr",
        "second-asr": "backfill-second-asr",
        "backfill_second_asr": "backfill-second-asr",
    }
    requested: list[str] = []
    invalid: list[str] = []
    for raw in stages:
        stage = str(raw).strip().lower().replace("_", "-")
        if not stage:
            continue
        stage = aliases.get(stage, stage)
        if stage not in AVAILABLE_PIPELINE_STAGE_ORDER:
            invalid.append(stage)
            continue
        if stage not in requested:
            requested.append(stage)
    if invalid:
        raise RuntimeError(
            "Unsupported worker stages: " + ", ".join(invalid)
        )
    return [stage for stage in AVAILABLE_PIPELINE_STAGE_ORDER if stage in requested]


def run_worker(
    settings: Settings,
    *,
    stage_limit: int,
    once: bool,
    stages: Optional[list[str]] = None,
    poll_sec: int | None = None,
    max_idle_cycles: int | None = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    with controlled_worker_parent_lifeline(settings):
        return _run_worker_with_lifeline(
            settings,
            stage_limit=stage_limit,
            once=once,
            stages=stages,
            poll_sec=poll_sec,
            max_idle_cycles=max_idle_cycles,
            progress_callback=progress_callback,
        )


def _run_worker_with_lifeline(
    settings: Settings,
    *,
    stage_limit: int,
    once: bool,
    stages: Optional[list[str]] = None,
    poll_sec: int | None = None,
    max_idle_cycles: int | None = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    session_factory = build_session_factory(settings)
    selected_stages = normalize_pipeline_stages(stages)
    enforce_controlled_worker_stages(
        settings,
        selected_stages,
        stage_limit=stage_limit,
    )
    poll_interval = max(1, poll_sec if poll_sec is not None else settings.worker_poll_sec)
    max_idle = (
        max_idle_cycles
        if max_idle_cycles is not None
        else max(0, settings.worker_max_idle_cycles)
    )
    primary_only_settings = replace(
        settings,
        dual_transcribe_enabled=False,
        secondary_transcribe_provider=None,
    )
    primary_transcribe_service = (
        TranscribeService(primary_only_settings) if "transcribe" in selected_stages else None
    )
    secondary_transcribe_service = (
        TranscribeService(settings) if "backfill-second-asr" in selected_stages else None
    )

    cycles = 0
    idle_cycles = 0
    totals = {
        stage: {"processed": 0, "success": 0, "failed": 0} for stage in selected_stages
    }
    runtime_receipts: Dict[str, Dict[str, Any]] = {
        stage: {
            "provider_invocations": {},
            "mlx_cache_release_attempts": 0,
            "mlx_cache_release_successes": 0,
        }
        for stage in selected_stages
    }
    while True:
        cycles += 1
        cycle_payload: Dict[str, Any] = {
            "cycle": cycles,
            "stages": list(selected_stages),
        }
        if "transcribe" in selected_stages:
            assert primary_transcribe_service is not None
            with session_factory() as session:
                cycle_payload["transcribe"] = primary_transcribe_service.run(
                    session,
                    limit=stage_limit,
                )
        if "backfill-second-asr" in selected_stages:
            assert secondary_transcribe_service is not None
            with session_factory() as session:
                cycle_payload["backfill-second-asr"] = secondary_transcribe_service.backfill_secondary_asr(
                    session,
                    limit=stage_limit,
                )
        if "resolve" in selected_stages:
            with session_factory() as session:
                cycle_payload["resolve"] = ResolveService(settings).run(session, limit=stage_limit)
        if "analyze" in selected_stages:
            with session_factory() as session:
                cycle_payload["analyze"] = AnalyzeService(settings).run(session, limit=stage_limit)
        if "sync" in selected_stages:
            with session_factory() as session:
                cycle_payload["sync"] = AmoCRMSyncService(settings).run(session, limit=stage_limit)

        if progress_callback is not None:
            progress_callback(cycle_payload)
        for stage in selected_stages:
            stage_result = cycle_payload.get(stage) or {}
            totals[stage]["processed"] += int(stage_result.get("processed", 0))
            totals[stage]["success"] += int(stage_result.get("success", 0))
            totals[stage]["failed"] += int(stage_result.get("failed", 0))
            receipt = stage_result.get("runtime_receipt")
            if isinstance(receipt, dict):
                providers = receipt.get("provider_invocations")
                if isinstance(providers, dict):
                    aggregate = runtime_receipts[stage]["provider_invocations"]
                    for provider, count in providers.items():
                        key = str(provider)
                        aggregate[key] = int(aggregate.get(key, 0)) + int(count or 0)
                for key in (
                    "mlx_cache_release_attempts",
                    "mlx_cache_release_successes",
                ):
                    runtime_receipts[stage][key] += int(receipt.get(key) or 0)

        cycle_work = sum(
            int((cycle_payload.get(stage) or {}).get("processed", 0))
            for stage in selected_stages
        )
        if once:
            return {
                "ok": True,
                "mode": "once",
                "cycles": cycles,
                "idle_cycles": idle_cycles,
                "totals": totals,
                "runtime_receipts": runtime_receipts,
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
                    "runtime_receipts": runtime_receipts,
                    "last_cycle": cycle_payload,
                    "stop_reason": "max_idle_cycles_reached",
                }
            time.sleep(poll_interval)
        else:
            idle_cycles = 0
