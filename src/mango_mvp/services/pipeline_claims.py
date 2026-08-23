from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

from mango_mvp.config import Settings
from mango_mvp.services.controlled_call_scope import require_unique_controlled_call

STAGE_WORKER_ID_ENV = "MANGO_CALLS_STAGE_WORKER_ID"
STAGE_WORKER_ID_RE = re.compile(r"(?:tr|bf|rs|an)-[0-9a-f]{32}")


def stage_worker_id_is_valid(prefix: str, worker_id: str) -> bool:
    return bool(
        STAGE_WORKER_ID_RE.fullmatch(worker_id)
        and worker_id.startswith(f"{prefix}-")
    )


def configured_stage_worker_id(prefix: str) -> str | None:
    """Return the orchestrator-owned claim identity, rejecting any mismatch."""
    worker_id = os.getenv(STAGE_WORKER_ID_ENV, "").strip()
    if not worker_id:
        return None
    if not stage_worker_id_is_valid(prefix, worker_id):
        raise RuntimeError(f"{STAGE_WORKER_ID_ENV} is invalid for stage {prefix}")
    return worker_id


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def lease_cutoff(settings: Settings, now: datetime | None = None) -> datetime:
    current = now or utc_now()
    timeout_sec = max(60, int(settings.pipeline_lease_timeout_sec))
    return current - timedelta(seconds=timeout_sec)


def release_stale_pipeline_claims(session: Session, settings: Settings, now: datetime | None = None) -> int:
    current = now or utc_now()
    cutoff = lease_cutoff(settings, current)
    scope = require_unique_controlled_call(session, settings)
    scope_sql = (
        " AND source_call_id = :controlled_source_call_id" if scope else ""
    )
    params = {"now": current, "cutoff": cutoff}
    if scope:
        params["controlled_source_call_id"] = scope.source_call_id
    total = 0
    total += int(
        session.execute(
            text(
                f"""
                UPDATE call_records
                   SET transcription_status = 'pending',
                       pipeline_stage = NULL,
                       pipeline_worker_id = NULL,
                       pipeline_claimed_at = NULL,
                       updated_at = :now
                 WHERE pipeline_stage = 'transcribe'
                   AND transcription_status = 'in_progress'
                   AND (
                        pipeline_claimed_at IS NULL
                        OR pipeline_claimed_at <= :cutoff
                   )
                   {scope_sql}
                """
            ),
            params,
        ).rowcount
        or 0
    )
    total += int(
        session.execute(
            text(
                f"""
                UPDATE call_records
                   SET resolve_status = 'pending',
                       pipeline_stage = NULL,
                       pipeline_worker_id = NULL,
                       pipeline_claimed_at = NULL,
                       updated_at = :now
                 WHERE pipeline_stage = 'resolve'
                   AND resolve_status = 'in_progress'
                   AND (
                        pipeline_claimed_at IS NULL
                        OR pipeline_claimed_at <= :cutoff
                   )
                   {scope_sql}
                """
            ),
            params,
        ).rowcount
        or 0
    )
    total += int(
        session.execute(
            text(
                f"""
                UPDATE call_records
                   SET pipeline_stage = NULL,
                       pipeline_worker_id = NULL,
                       pipeline_claimed_at = NULL,
                       updated_at = :now
                 WHERE pipeline_stage = 'backfill-second-asr'
                   AND (
                        pipeline_claimed_at IS NULL
                        OR pipeline_claimed_at <= :cutoff
                   )
                   {scope_sql}
                """
            ),
            params,
        ).rowcount
        or 0
    )
    # Older/interrupted runtimes may leave lease columns behind after the
    # corresponding stage status has already advanced.  Once such an orphan
    # is stale, no stage-specific recovery query above can own it, so clear
    # only the lease metadata and preserve all completed stage results.
    total += int(
        session.execute(
            text(
                f"""
                UPDATE call_records
                   SET pipeline_stage = NULL,
                       pipeline_worker_id = NULL,
                       pipeline_claimed_at = NULL,
                       updated_at = :now
                 WHERE (
                        pipeline_stage IS NOT NULL
                        OR pipeline_worker_id IS NOT NULL
                        OR pipeline_claimed_at IS NOT NULL
                   )
                   AND (
                        pipeline_claimed_at IS NULL
                        OR pipeline_claimed_at <= :cutoff
                   )
                   AND COALESCE(NOT (
                        (pipeline_stage = 'transcribe'
                         AND transcription_status = 'in_progress')
                        OR (pipeline_stage = 'resolve'
                            AND resolve_status = 'in_progress')
                        OR pipeline_stage = 'backfill-second-asr'
                   ), 1)
                   {scope_sql}
                """
            ),
            params,
        ).rowcount
        or 0
    )
    return total
