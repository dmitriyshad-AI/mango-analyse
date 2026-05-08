from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from mango_mvp.productization.capture import CaptureDecision, CapturePlanner, InMemorySeenCallStore
from mango_mvp.productization.contracts import TenantRef
from mango_mvp.productization.mango_office import MangoOfficePayloadMapper
from mango_mvp.productization.mango_office_client import (
    DEFAULT_MANGO_BASE_URL,
    MangoOfficeClient,
    MangoOfficeCredentials,
)
from mango_mvp.productization.payload_archive import write_shadow_poll_raw_payload_jsonl
from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


MANGO_LIVE_SHADOW_POLL_SCHEMA_VERSION = "mango_live_shadow_poll_v1"


class MangoPollClient(Protocol):
    def poll_call_history(self, since: datetime, until: datetime) -> Sequence[Mapping[str, Any]]:
        ...


@dataclass(frozen=True)
class MangoLiveShadowPollSummary:
    schema_version: str
    tenant_id: str
    product_db_path: str
    window_since: str
    window_until: str
    source_rows: int
    raw_payload_rows: int
    normalized_events: int
    normalization_errors: int
    seen_event_keys: int
    enqueue_shadow_capture: int
    skip_duplicate: int
    skip_no_recording: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_mango_live_shadow_poll_report(
    product_db_path: Path,
    product_root: Path,
    tenant_id: str,
    since: datetime,
    until: datetime,
    raw_payload_path: Path,
    base_url: str = DEFAULT_MANGO_BASE_URL,
    api_key: Optional[str] = None,
    api_salt: Optional[str] = None,
    allow_metadata_only: bool = False,
    include_job_run_seen: bool = True,
    client: Optional[MangoPollClient] = None,
) -> Mapping[str, Any]:
    product_db_path = product_db_path.resolve(strict=False)
    product_root = product_root.resolve(strict=False)
    raw_payload_path = raw_payload_path.resolve(strict=False)
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    guard_live_shadow_path(raw_payload_path, product_root, "raw payload")
    if until <= since:
        raise ValueError("until must be later than since")
    tenant = TenantRef(tenant_id)

    poll_client = client or build_env_mango_client(base_url=base_url, api_key=api_key, api_salt=api_salt)
    rows = tuple(poll_client.poll_call_history(since=since, until=until))
    raw_payload_rows = write_shadow_poll_raw_payload_jsonl(
        rows=rows,
        out_path=raw_payload_path,
        tenant_id=tenant.tenant_id,
        provider="mango",
        base_url=base_url,
        since=since.isoformat(),
        until=until.isoformat(),
    )

    mapper = MangoOfficePayloadMapper()
    events = []
    errors = []
    for index, row in enumerate(rows):
        try:
            events.append(mapper.from_payload(tenant=tenant, payload=row))
        except Exception as exc:
            errors.append({"index": index, "error": str(exc), "payload": dict(row)})

    seen_keys = read_seen_event_keys(product_db_path, include_job_runs=include_job_run_seen)
    planner = CapturePlanner(
        seen_store=InMemorySeenCallStore(seen_keys),
        require_recording=not allow_metadata_only,
    )
    decisions = planner.plan_batch(events)
    counts = summarize_decisions(decisions)
    blocked = len(errors)
    warnings = counts["SKIP_NO_RECORDING"]
    summary = MangoLiveShadowPollSummary(
        schema_version=MANGO_LIVE_SHADOW_POLL_SCHEMA_VERSION,
        tenant_id=tenant.tenant_id,
        product_db_path=str(product_db_path),
        window_since=since.isoformat(),
        window_until=until.isoformat(),
        source_rows=len(rows),
        raw_payload_rows=raw_payload_rows,
        normalized_events=len(events),
        normalization_errors=len(errors),
        seen_event_keys=len(seen_keys),
        enqueue_shadow_capture=counts["ENQUEUE_SHADOW_CAPTURE"],
        skip_duplicate=counts["SKIP_DUPLICATE"],
        skip_no_recording=counts["SKIP_NO_RECORDING"],
        validation_ok=blocked == 0,
        blocked=blocked,
        warnings=warnings,
    )
    return {
        "validation_ok": summary.validation_ok,
        "summary": summary.to_json_dict(),
        "window": {"since": since.isoformat(), "until": until.isoformat()},
        "source": {
            "provider": "mango",
            "base_url": base_url,
            "credentials_ref": {
                "api_key": "env:MANGO_OFFICE_API_KEY",
                "api_salt": "env:MANGO_OFFICE_API_SALT",
            },
        },
        "raw_payload_archive": {"path": str(raw_payload_path), "rows": raw_payload_rows},
        "action_counts": counts,
        "decisions": [decision_to_dict(decision) for decision in decisions],
        "normalization_errors": errors,
        "safety": {
            "download_audio": False,
            "run_asr": False,
            "run_ra": False,
            "write_crm": False,
            "write_runtime_db": False,
            "stable_runtime_writes": False,
        },
    }


def build_env_mango_client(
    base_url: str = DEFAULT_MANGO_BASE_URL,
    api_key: Optional[str] = None,
    api_salt: Optional[str] = None,
) -> MangoOfficeClient:
    load_env_file()
    resolved_key = api_key or os.getenv("MANGO_OFFICE_API_KEY")
    resolved_salt = api_salt or os.getenv("MANGO_OFFICE_API_SALT")
    if not resolved_key or not resolved_salt:
        raise ValueError("MANGO_OFFICE_API_KEY and MANGO_OFFICE_API_SALT are required")
    return MangoOfficeClient(
        credentials=MangoOfficeCredentials(api_key=resolved_key, api_salt=resolved_salt),
        base_url=base_url or os.getenv("MANGO_OFFICE_BASE_URL", DEFAULT_MANGO_BASE_URL),
    )


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def read_seen_event_keys(product_db_path: Path, include_job_runs: bool = True) -> set[str]:
    keys: set[str] = set()
    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT event_key FROM product_calls WHERE event_key IS NOT NULL AND event_key != ''").fetchall()
        keys.update(clean(row["event_key"]) for row in rows if clean(row["event_key"]))
        if include_job_runs and relation_exists(con, "job_runs"):
            job_rows = con.execute(
                """
                SELECT result_json
                  FROM job_runs
                 WHERE job_type = 'shadow_poll'
                   AND status = 'succeeded'
                   AND result_json IS NOT NULL
                   AND result_json != ''
                """
            ).fetchall()
            for row in job_rows:
                keys.update(event_keys_from_job_result(clean(row["result_json"])))
    return keys


def event_keys_from_job_result(result_json: str) -> set[str]:
    if not result_json:
        return set()
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return set()
    if not isinstance(data, Mapping):
        return set()
    keys: set[str] = set()
    for decision in data.get("decisions") or ():
        if not isinstance(decision, Mapping):
            continue
        action_code = clean(decision.get("action_code")).upper()
        action = clean(decision.get("action"))
        if action_code != "ENQUEUE_SHADOW_CAPTURE" and action != "enqueue_shadow_capture":
            continue
        event = decision.get("event")
        if isinstance(event, Mapping):
            event_key = clean(event.get("event_key"))
            if event_key:
                keys.add(event_key)
    return keys


def summarize_decisions(decisions: Sequence[CaptureDecision]) -> Mapping[str, int]:
    counts = {
        "ENQUEUE_SHADOW_CAPTURE": 0,
        "SKIP_DUPLICATE": 0,
        "SKIP_NO_RECORDING": 0,
    }
    for decision in decisions:
        counts[decision.action.name] = counts.get(decision.action.name, 0) + 1
    return counts


def decision_to_dict(decision: CaptureDecision) -> Mapping[str, Any]:
    candidate = asdict(decision.candidate) if decision.candidate else None
    if candidate is not None and decision.candidate is not None:
        candidate["started_at"] = decision.candidate.started_at.isoformat()
        candidate["direction"] = decision.candidate.direction.value
    return {
        "action": decision.action.value,
        "action_code": decision.action.name,
        "reason": decision.reason,
        "event": {
            "event_key": decision.event.event_key,
            "provider_call_id": decision.event.provider_call_id,
            "started_at": decision.event.started_at.isoformat(),
            "ended_at": decision.event.ended_at.isoformat() if decision.event.ended_at else None,
            "duration_seconds": decision.event.duration_seconds,
            "direction": decision.event.direction.value,
            "client_phone": decision.event.client_phone,
            "manager_ref": decision.event.manager_ref,
            "recording_ref": decision.event.recording_ref,
            "recording_url": decision.event.recording_url,
        },
        "candidate": candidate,
    }


def relation_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute("select 1 from sqlite_master where type = 'table' and name = ?", (name,)).fetchone()
    return row is not None


def guard_live_shadow_path(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")
