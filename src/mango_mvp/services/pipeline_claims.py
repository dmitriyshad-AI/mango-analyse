from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

from mango_mvp.config import Settings


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def lease_cutoff(settings: Settings, now: datetime | None = None) -> datetime:
    current = now or utc_now()
    timeout_sec = max(60, int(settings.pipeline_lease_timeout_sec))
    return current - timedelta(seconds=timeout_sec)


def release_stale_pipeline_claims(session: Session, settings: Settings, now: datetime | None = None) -> int:
    current = now or utc_now()
    cutoff = lease_cutoff(settings, current)
    total = 0
    total += int(
        session.execute(
            text(
                """
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
                """
            ),
            {"now": current, "cutoff": cutoff},
        ).rowcount
        or 0
    )
    total += int(
        session.execute(
            text(
                """
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
                """
            ),
            {"now": current, "cutoff": cutoff},
        ).rowcount
        or 0
    )
    total += int(
        session.execute(
            text(
                """
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
                """
            ),
            {"now": current, "cutoff": cutoff},
        ).rowcount
        or 0
    )
    return total

